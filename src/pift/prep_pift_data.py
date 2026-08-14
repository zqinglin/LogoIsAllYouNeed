#!/usr/bin/env python
"""Prepare data for the PIFT (Patch-Invariant Fine-Tuning) experiment.

Produces three artifacts from VideoFeedback (train_regression.json):
  test_clean.json      : held-out clean videos with human labels  -> accuracy eval
  train_contam.json    : Sora watermark on HIGH-VQ clips (patch <-> quality correlated)
                         reproduces the inflation  -> contaminated baseline
  train_pift.json      : DIVERSE patch (sora / mirrored / gray box / random rect,
                         random position) on a VQ-BALANCED subset (patch _|_ quality)
                         -> our patch-invariant training set

Same base clips, same labels, same number of patched clips per training set;
only the patch<->quality correlation (and patch diversity) differs.
"""
import os, json, argparse, shutil, random
from PIL import Image, ImageOps, ImageDraw

VQ_KEYS = ["visual quality", "visual_quality", "Visual Quality"]


def vq_of(labels):
    for k in VQ_KEYS:
        if k in labels:
            return float(labels[k])
    for v in labels.values():
        try:
            return float(v)
        except Exception:
            pass
    return None


def load_sora_wm(path, scale_w=160):
    src = Image.open(path).convert("RGB")
    gray = src.convert("L")
    mask = gray.point(lambda x: 255 if x > 45 else 0)
    bbox = mask.getbbox()
    if bbox:
        x0, y0, x1, y1 = bbox
        pad = 6
        x0 = max(0, x0 - pad); y0 = max(0, y0 - pad)
        x1 = min(src.size[0], x1 + pad); y1 = min(src.size[1], y1 + pad)
        src = src.crop((x0, y0, x1, y1)); gray = gray.crop((x0, y0, x1, y1))
    wm = src.convert("RGBA")
    alpha = gray.point(lambda x: 0 if x < 45 else min(255, int(x * 1.15)))
    wm.putalpha(alpha)
    w, h = wm.size
    return wm.resize((scale_w, max(1, int(h * scale_w / w))), Image.LANCZOS)


def make_gray_box(w=150, h=90, val=128, alpha=210):
    box = Image.new("RGBA", (w, h), (val, val, val, alpha))
    return box


def make_random_rect(rng, w=150, h=90):
    c = (rng.randint(60, 200), rng.randint(60, 200), rng.randint(60, 200), 210)
    return Image.new("RGBA", (w, h), c)


def patch_variant(kind, sora_wm, mirrored_wm, rng):
    if kind == "sora":
        return sora_wm
    if kind == "mirrored":
        return mirrored_wm
    if kind == "gray":
        return make_gray_box()
    return make_random_rect(rng)


def apply_patch(frame_path, out_path, patch, pos):
    img = Image.open(frame_path).convert("RGB")
    W, H = img.size
    p = patch
    if p.size[0] > W // 3:
        nw = W // 4
        p = p.resize((nw, max(1, int(p.size[1] * nw / p.size[0]))), Image.LANCZOS)
    ww, wh = p.size
    m = max(4, W // 50)
    xy = {"br": (W - ww - m, H - wh - m), "tl": (m, m),
          "tr": (W - ww - m, m), "bl": (m, H - wh - m)}[pos]
    base = img.convert("RGBA")
    base.alpha_composite(p, xy)
    base.convert("RGB").save(out_path, quality=95)


def copy_clean(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy(src, dst)


def cap_frames(imgs, mf):
    if len(imgs) <= mf:
        return imgs
    idx = [round(i * (len(imgs) - 1) / (mf - 1)) for i in range(mf)]
    return [imgs[j] for j in idx]


def write_clean(clips, images_root, out_img_root, tag, mf):
    out = []
    for c in clips:
        imgs = cap_frames(c["images"], mf)
        new = []
        for rel in imgs:
            src = os.path.join(images_root, rel) if not os.path.isabs(rel) else rel
            sub = rel.split("images/", 1)[-1]
            dst = os.path.join(out_img_root, sub)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if not os.path.exists(dst):
                copy_clean(src, dst)
            new.append(os.path.join(tag + "_images", sub))
        out.append({"id": c["id"], "images": new, "prompt": c["prompt"], "labels": c["labels"]})
    return out


def materialize(clips, wm_ids, images_root, out_img_root, tag, mf, mode, sora_wm, mirrored_wm, seed):
    """mode='contam' -> always sora watermark; mode='pift' -> diverse random patch."""
    positions = ["br", "tl", "tr", "bl"]
    kinds = ["sora", "mirrored", "gray", "rect"]
    out = []
    for i, c in enumerate(clips):
        rng = random.Random(seed + hash(str(c["id"])) % 100000)
        is_wm = c["id"] in wm_ids
        imgs = cap_frames(c["images"], mf)
        new = []
        for j, rel in enumerate(imgs):
            src = os.path.join(images_root, rel) if not os.path.isabs(rel) else rel
            sub = rel.split("images/", 1)[-1]
            dst = os.path.join(out_img_root, sub)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if is_wm:
                if mode == "contam":
                    patch = sora_wm; pos = positions[(i + j) % 4]
                else:  # pift: diverse patch, fixed per-clip type, jittered position
                    kind = kinds[rng.randint(0, 3)]
                    patch = patch_variant(kind, sora_wm, mirrored_wm, rng)
                    pos = positions[rng.randint(0, 3)]
                apply_patch(src, dst, patch, pos)
            else:
                if not os.path.exists(dst):
                    copy_clean(src, dst)
            new.append(os.path.join(tag + "_images", sub))
        out.append({"id": c["id"], "images": new, "prompt": c["prompt"], "labels": c["labels"]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-json", required=True)
    ap.add_argument("--images-root", required=True)
    ap.add_argument("--sora-wm", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-train", type=int, default=12000)
    ap.add_argument("--n-test", type=int, default=2500)
    ap.add_argument("--vq-high", type=float, default=3.0)
    ap.add_argument("--max-frames", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    data = json.load(open(args.train_json))
    base = [c for c in data if not str(c["id"]).startswith("s") and vq_of(c["labels"]) is not None]
    base = sorted(base, key=lambda c: str(c["id"]))
    rng = random.Random(args.seed)
    rng.shuffle(base)

    # stratified-ish held-out clean test, then train subset from the rest
    test = base[: args.n_test]
    train = base[args.n_test: args.n_test + args.n_train]
    print(f"[split] base={len(base)}  test={len(test)}  train={len(train)}")

    os.makedirs(args.out, exist_ok=True)
    mf = args.max_frames
    sora_wm = load_sora_wm(args.sora_wm, scale_w=160)
    mirrored_wm = ImageOps.mirror(sora_wm)

    # test (clean)
    test_out = write_clean(test, args.images_root, os.path.join(args.out, "test_images"), "test", mf)
    json.dump(test_out, open(os.path.join(args.out, "test_clean.json"), "w"))
    print(f"[test] wrote {len(test_out)} clean -> test_clean.json")

    # which train clips get patched: k = #(VQ>=vq_high)
    contam_wm = {c["id"] for c in train if vq_of(c["labels"]) >= args.vq_high}
    k = len(contam_wm)
    by_vq = sorted(train, key=lambda c: (vq_of(c["labels"]), str(c["id"])))
    step = max(1, len(by_vq) // max(1, k))
    pift_wm = set(list({by_vq[i]["id"] for i in range(0, len(by_vq), step)})[:k])
    print(f"[patch] contam(high-VQ)={k}  pift(VQ-balanced)={len(pift_wm)}")

    contam = materialize(train, contam_wm, args.images_root, os.path.join(args.out, "contam_images"),
                         "contam", mf, "contam", sora_wm, mirrored_wm, args.seed)
    json.dump(contam, open(os.path.join(args.out, "train_contam.json"), "w"))
    print(f"[contam] wrote {len(contam)} -> train_contam.json")

    pift = materialize(train, pift_wm, args.images_root, os.path.join(args.out, "pift_images"),
                       "pift", mf, "pift", sora_wm, mirrored_wm, args.seed)
    json.dump(pift, open(os.path.join(args.out, "train_pift.json"), "w"))
    print(f"[pift] wrote {len(pift)} -> train_pift.json")

    def mean_vq(clips, ids):
        w = [vq_of(c["labels"]) for c in clips if c["id"] in ids]
        nw = [vq_of(c["labels"]) for c in clips if c["id"] not in ids]
        return (sum(w) / len(w) if w else 0, sum(nw) / len(nw) if nw else 0)
    cw, cn = mean_vq(train, contam_wm)
    pw, pn = mean_vq(train, pift_wm)
    print(f"[check] contam meanVQ patched={cw:.2f} vs clean={cn:.2f}  (should DIFFER -> correlated)")
    print(f"[check] pift   meanVQ patched={pw:.2f} vs clean={pn:.2f}  (should MATCH -> decorrelated)")


if __name__ == "__main__":
    main()
