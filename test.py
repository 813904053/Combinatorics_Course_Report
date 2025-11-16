import cv2

# 方法1：查看所有已编译的扩展模块
print("已编译的扩展模块：")
for mod in cv2.getBuildInformation().split('\n'):
    if 'freetype' in mod.lower() or 'xfeatures2d' in mod.lower():
        print(mod.strip())

# 方法2：直接尝试导入 freetype（若报错则未支持）
try:
    from cv2 import freetype
    print("✅ freetype 模块可用！")
except ImportError as e:
    print("❌ 未找到 freetype 模块：", e)