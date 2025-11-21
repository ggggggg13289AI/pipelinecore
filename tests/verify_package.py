#!/usr/bin/env python3
"""验证脚本：检查包结构和资源文件配置"""

import zipfile
from pathlib import Path

def verify_package_structure():
    """验证构建的包中包含正确的资源文件"""
    print("🔍 验证包结构和资源文件配置\n")

    # 1. 检查源代码中的 resource 目录
    print("1️⃣ 检查源代码结构:")
    src_resource = Path("../src/pipelinecore/resource")
    if src_resource.exists():
        print(f"   ✅ resource 目录存在: {src_resource}")
        files = list(src_resource.iterdir())
        print(f"   ✅ 包含 {len(files)} 个文件:")
        for file in files:
            print(f"      - {file.name} ({file.stat().st_size} bytes)")
    else:
        print(f"   ❌ resource 目录不存在")
        return False

    # 2. 检查构建的 wheel 包
    print("\n2️⃣ 检查构建的 wheel 包:")
    dist_dir = Path("../dist")
    wheel_files = list(dist_dir.glob("*.whl")) if dist_dir.exists() else []

    if not wheel_files:
        print("   ⚠️  没有找到 wheel 包，请先运行: uv builder")
        return False

    wheel_path = wheel_files[0]
    print(f"   ✅ 找到 wheel 包: {wheel_path.name}")

    # 3. 检查 wheel 包内容
    print("\n3️⃣ 检查 wheel 包内容:")
    with zipfile.ZipFile(wheel_path, 'r') as zf:
        files = zf.namelist()
        resource_files = [f for f in files if 'resource/' in f]

        if resource_files:
            print(f"   ✅ wheel 包包含 {len(resource_files)} 个资源文件:")
            for file in resource_files:
                info = zf.getinfo(file)
                print(f"      - {file} ({info.file_size} bytes)")
        else:
            print("   ❌ wheel 包中没有找到 resource 文件")
            return False

    # 4. 显示安装后的路径
    print("\n4️⃣ 安装后的文件路径:")
    print("   安装包后，资源文件将位于:")
    print("   <site-packages>/pipelinecore/resource/")
    print("\n   在代码中访问资源文件的方式:")
    print("   ```python")
    print("   from pathlib import Path")
    print("   import pipelinecore")
    print("   ")
    print("   # 方式 1: 通过包路径")
    print("   pkg_path = Path(pipelinecore.__file__).parent")
    print("   resource_path = pkg_path / 'resource'")
    print("   ")
    print("   # 方式 2: 使用 importlib.resources (Python 3.9+)")
    print("   from importlib.resources import files")
    print("   resource_path = files('pipelinecore') / 'resource'")
    print("   ```")

    print("\n🎉 验证完成！资源文件配置正确。")
    return True

if __name__ == "__main__":
    import sys
    success = verify_package_structure()
    sys.exit(0 if success else 1)
