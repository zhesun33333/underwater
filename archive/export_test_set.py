"""
导出筛选后的测试集为独立压缩包。

输出结构:
  testset_export/
    sft_test_highquality.jsonl   ← 内嵌 _gt 标签, audio 路径相对于本目录
    audio/                       ← WAV 文件, 保持原始目录结构

用于 testsite 评估:
  python scripts/run_eval.py --data testset_export/sft_test_highquality.jsonl --audio-root testset_export

用法:
  python export_test_set.py
  python export_test_set.py --no-compress   # 纯 tar, 最快
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

PROCESSED_ROOT = Path("processed_audio")
AUDIO_EXT = ".wav"


def main():
    parser = argparse.ArgumentParser(description="导出测试集 (JSONL + WAV)")
    parser.add_argument("--input", default="dataset/sft_test_highquality.jsonl")
    parser.add_argument("--output", default="testset_export")
    parser.add_argument("--no-compress", action="store_true",
                        help="不打压缩, 只打包为 .tar (最快)")
    args = parser.parse_args()

    jsonl_path = Path(args.input)
    if not jsonl_path.exists():
        print(f"错误: {args.input} 不存在, 请先运行 filter_test_set.py")
        return 1

    out_dir = Path(args.output)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    # ---- 第一遍: 收集所有 WAV 文件, 重写 JSONL 路径 ----
    audio_dir = out_dir / "audio"
    records = []
    missing = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            arel = item.get("audio", "")
            if not arel:
                continue

            wav_src = PROCESSED_ROOT / arel
            if not wav_src.exists():
                print(f"  [WARN] WAV 缺失: {arel}")
                missing += 1
                continue

            # 重写 audio 路径: 相对于 out_dir (testset_export)
            item["audio"] = str(Path("audio") / arel)
            records.append((item, wav_src))

    # ---- 第二遍: 复制 WAV ----
    print(f"共 {len(records)} 个样本, 开始复制 WAV ...")

    for i, (item, wav_src) in enumerate(records):
        wav_dst = audio_dir / wav_src.relative_to(PROCESSED_ROOT)
        wav_dst.parent.mkdir(parents=True, exist_ok=True)
        if not wav_dst.exists():
            shutil.copy2(wav_src, wav_dst)

        if (i + 1) % 500 == 0:
            print(f"  已复制 {i + 1}/{len(records)} ...")

    # ---- 第三遍: 写 JSONL (带内嵌 _gt, 路径已调整为相对) ----
    output_jsonl = out_dir / jsonl_path.name
    with open(output_jsonl, "w", encoding="utf-8") as fout:
        for item, _ in records:
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")

    # ---- 第四遍: 打包 ----
    out_dir_abs = out_dir.resolve()
    ext = "tar" if args.no_compress else "tar.gz"
    archive_path = out_dir_abs.parent / f"{out_dir.name}.{ext}"
    print(f"\n打包: {archive_path} ...")

    tar_cmd = ["tar", "-cf", str(archive_path), "-C", str(out_dir_abs.parent), out_dir.name]
    if not args.no_compress:
        tar_cmd.insert(2, "-I")
        tar_cmd.insert(3, "pigz -p4" if shutil.which("pigz") else "gzip")

    result = subprocess.run(tar_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  tar 失败, 回退到 Python tarfile...")
        import tarfile
        mode = "w" if args.no_compress else "w:gz"
        with tarfile.open(archive_path, mode) as tf:
            tf.add(out_dir_abs, arcname=out_dir.name)

    archive_size_mb = archive_path.stat().st_size / (1024 * 1024)
    wav_total_mb = sum(wav_src.stat().st_size for _, wav_src in records) / (1024 * 1024)

    # ---- 汇总 ----
    print(f"\n{'='*50}")
    print(f"导出完成")
    print(f"  样本数:     {len(records)}")
    if missing > 0:
        print(f"  缺失 WAV:   {missing}")
    print(f"  WAV 大小:   {wav_total_mb:.1f} MB")
    print(f"  压缩包:     {archive_path.resolve()} ({archive_size_mb:.1f} MB)")
    print(f"")
    print(f"  testsite 用法:")
    print(f"    python scripts/run_eval.py \\")
    print(f"      --data {out_dir.name}/{output_jsonl.name} \\")
    print(f"      --audio-root {out_dir.name}")
    print(f"{'='*50}")


if __name__ == "__main__":
    sys.exit(main())
