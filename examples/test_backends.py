"""快速测试后端可用性"""

from hllm.backends import list_backends, get_backend_info

print("=" * 60)
print("HLLM 后端可用性测试")
print("=" * 60)

# 列出可用后端
backends = list_backends()
print(f"\n✅ 可用后端: {backends}")

# 详细信息
info = get_backend_info()
print("\n后端详情:")
for name, data in info.items():
    if data.get("available"):
        print(f"  📦 {name}:")
        print(f"     - 默认设备: {data.get('default_device', 'N/A')}")
        print(f"     - 支持量化: {data.get('supports_quantization', False)}")
    else:
        print(f"  ❌ {name}: 不可用")
        print(f"     - 错误: {data.get('error', 'Unknown')}")

print("\n" + "=" * 60)
print("测试完成!")
print("=" * 60)
