"""
Script tải mô hình NVIDIA GR00T-N1.7-3B và GEAR-SONIC từ Hugging Face.
Hỗ trợ tự động xử lý redirect CDN/LFS, xác thực token và resume tải file.
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request


class SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Handler xử lý redirect an toàn giữa Hugging Face và S3/CloudFront CDN.
    Khi chuyển hướng sang CDN có presigned URL, loại bỏ header Authorization
    để tránh lỗi xung đột xác thực (AWS S3 'Only one auth mechanism allowed').
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new_req = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new_req is not None:
            old_host = urllib.parse.urlparse(req.full_url).netloc
            new_host = urllib.parse.urlparse(newurl).netloc
            if old_host != new_host and "huggingface.co" not in new_host:
                new_req.headers.pop("Authorization", None)
                new_req.headers.pop("authorization", None)
        return new_req


def download_file(file_url: str, dest_path: str, token: str):
    opener = urllib.request.build_opener(SafeRedirectHandler())
    headers = {"Authorization": f"Bearer {token}", "User-Agent": "HF-GR00T-Downloader"}
    req = urllib.request.Request(file_url, headers=headers)

    # Resume download nếu file đã tải dở dang
    downloaded = 0
    if os.path.exists(dest_path):
        downloaded = os.path.getsize(dest_path)

    if downloaded > 0:
        headers["Range"] = f"bytes={downloaded}-"
        req = urllib.request.Request(file_url, headers=headers)

    try:
        with opener.open(req) as resp:
            content_range = resp.headers.get("Content-Range")
            content_length = resp.headers.get("Content-Length")
            
            if content_range:
                total_size = int(content_range.split("/")[-1])
                mode = "ab"
            elif resp.status == 200:
                total_size = int(content_length) if content_length else 0
                downloaded = 0
                mode = "wb"
            else:
                total_size = int(content_length) if content_length else 0
                mode = "ab"

            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            chunk_size = 2 * 1024 * 1024  # 2MB chunks
            with open(dest_path, mode) as f:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        pct = (downloaded / total_size) * 100
                        mb_done = downloaded / (1024 * 1024)
                        mb_total = total_size / (1024 * 1024)
                        sys.stdout.write(f"\r    [{pct:5.1f}%] {mb_done:7.1f} / {mb_total:7.1f} MB")
                        sys.stdout.flush()
        print()
        return True
    except urllib.error.HTTPError as e:
        if e.code == 416:  # Requested Range Not Satisfiable -> file đã tải đầy đủ
            print("    [100.0%] File đã hoàn tất trước đó.")
            return True
        raise e


def download_repo(repo_id: str, local_dir: str, token: str, exclude_patterns=None, include_patterns=None):
    print(f"\n=======================================================")
    print(f"  Bắt đầu tải kho mô hình: {repo_id}")
    print(f"  Thư mục đích           : {local_dir}")
    print(f"=======================================================")

    api_url = f"https://huggingface.co/api/models/{repo_id}"
    req = urllib.request.Request(api_url, headers={"Authorization": f"Bearer {token}"})

    try:
        with urllib.request.urlopen(req) as resp:
            repo_info = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        if e.code == 401:
            print(f"[LỖI 401] Token không hợp lệ hoặc tài khoản chưa được cấp quyền truy cập {repo_id}.")
            print("Vui lòng đảm bảo bạn đã bấm 'Request Access' tại https://huggingface.co/" + repo_id)
        elif e.code == 403:
            print(f"[LỖI 403] Quyền truy cập bị từ chối đối với repo {repo_id}.")
        else:
            print(f"[LỖI HTTP {e.code}] {e.reason}")
        return False

    siblings = repo_info.get("siblings", [])
    files = [s["rfilename"] for s in siblings if not s["rfilename"].startswith(".")]

    if exclude_patterns:
        excludes = [p.strip() for p in exclude_patterns.split(",") if p.strip()]
        files = [f for f in files if not any(f.startswith(e) or e in f for e in excludes)]

    if include_patterns:
        includes = [p.strip() for p in include_patterns.split(",") if p.strip()]
        files = [f for f in files if any(f.startswith(i) or i in f for i in includes)]

    print(f"  -> Sẽ tải {len(files)} files được chọn từ repository.")
    os.makedirs(local_dir, exist_ok=True)

    success_count = 0
    for idx, rfilename in enumerate(files, start=1):
        dest_path = os.path.join(local_dir, rfilename)
        file_url = f"https://huggingface.co/{repo_id}/resolve/main/{rfilename}"
        print(f"[{idx}/{len(files)}] Đang tải {rfilename}...")
        try:
            download_file(file_url, dest_path, token)
            success_count += 1
        except Exception as e:
            print(f"    [!] Thất bại khi tải {rfilename}: {e}")

    print(f"\nHoàn tất tải {success_count}/{len(files)} files cho {repo_id}.\n")
    return success_count == len(files)


def main():
    parser = argparse.ArgumentParser(description="Tải mô hình từ Hugging Face.")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face User Access Token (hf_...)")
    parser.add_argument("--model", type=str, default="nvidia/GR00T-N1.7-3B", help="Model Repo ID")
    parser.add_argument("--output-dir", type=str, default="./checkpoints/GR00T-N1.7-3B", help="Thư mục lưu")
    parser.add_argument("--exclude", type=str, default=None, help="Các pattern file cần bỏ qua (phân cách bằng dấu phẩy)")
    parser.add_argument("--include", type=str, default=None, help="Các pattern file cần tải (phân cách bằng dấu phẩy)")
    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        print("[LỖI] Thiếu Hugging Face Token! Vui lòng truyền qua tham số --token hoặc đặt biến môi trường HF_TOKEN.")
        sys.exit(1)

    ok = download_repo(args.model, args.output_dir, token, exclude_patterns=args.exclude, include_patterns=args.include)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

