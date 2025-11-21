#!/usr/bin/env python3
"""测试脚本：验证 resource 目录在安装后的位置"""

import sys
from pathlib import Path

def test_resource_location():
    """测试资源文件是否在正确的位置"""
    try:
        import pipelinecore

        # 获取 pipelinecore 包的安装路径
        package_path = Path(pipelinecore.__file__).parent
        print(f"✅ pipelinecore 包路径: {package_path}")

        # 检查 resource 目录
        resource_dir = package_path / "resource"
        if resource_dir.exists():
            print(f"✅ resource 目录存在: {resource_dir}")

            # 列出 resource 目录中的文件
            files = list(resource_dir.iterdir())
            print(f"✅ resource 目录包含 {len(files)} 个文件:")
            for file in files:
                print(f"   - {file.name} ({file.stat().st_size} bytes)")

            # 测试读取 DICOM 文件
            dicom_file = resource_dir / "SEG_20230210_160056_635_S3.dcm"
            if dicom_file.exists():
                print(f"✅ DICOM 文件可访问: {dicom_file}")
                print(f"   文件大小: {dicom_file.stat().st_size} bytes")
            else:
                print(f"❌ DICOM 文件不存在")
                return False
        else:
            print(f"❌ resource 目录不存在: {resource_dir}")
            return False

        print("\n🎉 所有测试通过！资源文件配置正确。")
        return True

    except ImportError as e:
        print(f"❌ 无法导入 pipelinecore: {e}")
        print("   请先安装包: uv pip install -e .")
        return False
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

if __name__ == "__main__":
    success = test_resource_location()
    sys.exit(0 if success else 1)
