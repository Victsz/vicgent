# GOTYOU
import base64
from pathlib import Path
from langchain_core.tools import tool
import os
import io
from typing import List
import fitz  # PyMuPDF
from PIL import Image
def pdf_to_merged_image(pdf_path: str, dpi: int = 144) -> bytes:
    """
    将PDF文件的每一页转换为高质量图片，然后垂直合并为一张大图
    
    Args:
        pdf_path: PDF文件路径
        dpi: 图像分辨率，默认144
        
    Returns:
        bytes: 合并后图片的字节数据
    """
    images = []
    
    # 打开PDF文档
    pdf_document = fitz.open(pdf_path)
    
    # 计算缩放比例
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    
    # 遍历每一页
    for page_num in range(pdf_document.page_count):
        page = pdf_document[page_num]
        
        # 获取页面像素图
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        
        # 转换为PIL Image
        img_data = pixmap.tobytes("png")
        img = Image.open(io.BytesIO(img_data))
        
        # 如果有透明通道，转换为RGB
        if img.mode in ('RGBA', 'LA'):
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
            img = background
            
        images.append(img)
    
    pdf_document.close()
    
    # 垂直合并所有图片
    if not images:
        raise ValueError("PDF文件没有页面或无法读取")
    
    # 计算合并后的尺寸
    total_width = max(img.width for img in images)
    total_height = sum(img.height for img in images)
    
    # 创建合并后的图片
    merged_image = Image.new('RGB', (total_width, total_height), (255, 255, 255))
    
    # 粘贴每张图片
    y_offset = 0
    for img in images:
        # 居中粘贴
        x_offset = (total_width - img.width) // 2
        merged_image.paste(img, (x_offset, y_offset))
        y_offset += img.height
    
    # 转换为字节数据
    img_byte_arr = io.BytesIO()
    merged_image.save(img_byte_arr, format='PNG')
    return img_byte_arr.getvalue()

def load_image(filename: str, table = True) -> str:
    file_path = Path(filename)
    
    # 检查文件类型
    if file_path.suffix.lower() == '.pdf':
        # 处理PDF文件
        image_data = pdf_to_merged_image(filename)
        mime_type = "image/png"
        # 保存为PNG文件供肉眼查看
        output_path = file_path.parent / "processed" / f"{file_path.stem}.png"
        with open(output_path, "wb") as f:
            f.write(image_data)
    else:
        # 处理普通图片文件
        image_data = file_path.read_bytes()
        mime_type = "image/jpeg"
    
    # 转换为base64
    image_b64 = base64.b64encode(image_data).decode("utf-8")
    
    prompt = "Extract the table/s from this image output as markdown."
    if not table:
        prompt = "请帮忙提取图片"
    
    content = [
            {
                "type": "text",
                "text": prompt,
            },
            {
                "type": "image",
                "source_type": "base64",
                "data": image_b64,
                "mime_type": mime_type,
            },
        ]
    return content

@tool
def save_markdown_table(table: str, filename: str) -> None:
    """Save a markdown table to a file."""
    try:
        filename = filename.strip()
        folder = os.environ.get("OUTPUT_FOLDER","/home/victor/workspace/playgrounds/langchain/test_data/image_table")
        output = Path(folder)/filename
        with output.open("w") as f:
            f.write(table)
        return f"Saved markdown table to {output}"
    except Exception as e:
        return f"Error saving markdown table {output}: {e}"