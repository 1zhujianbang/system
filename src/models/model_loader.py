import torch
from ..config.config_manager import ModelConfig
from pathlib import Path
import os

def check_model_file(file_path):
    """检查模型文件完整性"""
    file_path = Path(file_path)
    
    print(f"🔍 检查模型文件: {file_path}")
    print(f"📊 文件大小: {file_path.stat().st_size} 字节")
    
    # 检查文件是否为空
    if file_path.stat().st_size == 0:
        print("❌ 模型文件为空!")
        return False
    
    # 尝试读取文件头信息
    try:
        with open(file_path, 'rb') as f:
            # PyTorch 文件通常以特定的魔术数字开头
            header = f.read(8)
            print(f"🔍 文件头: {header.hex()}")
    except Exception as e:
        print(f"❌ 读取文件头失败: {e}")
        return False
    
    return True

class ModelLoader:
    def __init__(self, models_dir: str = "./models"):
        self.models_dir = Path(models_dir)
        print(f"📁 模型目录路径: {self.models_dir.absolute()}")
        
        # 检查目录是否存在
        if not self.models_dir.exists():
            print(f"⚠️  模型目录不存在: {self.models_dir}")
            # 尝试创建目录
            self.models_dir.mkdir(parents=True, exist_ok=True)
            print(f"📁 已创建模型目录: {self.models_dir}")

    def load_model(self, model_config: ModelConfig):
        # 添加详细的调试信息
        print(f"🔍 开始加载模型，配置: {model_config}")
        
        if model_config is None:
            raise ValueError("model_config 不能为 None")
        
        if not hasattr(model_config, 'model_name') or not model_config.model_name:
            raise ValueError(f"model_config 必须包含有效的 model_name，当前: {getattr(model_config, 'model_name', 'None')}")
        
        model_path = self.models_dir / f"{model_config.model_name}"
        print(f"🔍 模型完整路径: {model_path.absolute()}")
        
        # 检查文件是否存在
        if not model_path.exists():
            # 列出目录内容帮助调试
            print(f"📁 模型目录内容:")
            try:
                for file in self.models_dir.iterdir():
                    print(f"   - {file.name}")
            except Exception as e:
                print(f"   无法读取目录: {e}")
            
            raise FileNotFoundError(f"模型文件 {model_path} 不存在")

        print(f"✅ 找到模型文件: {model_path}")
        
        # 检查文件完整性
        if not check_model_file(model_path):
            raise ValueError("模型文件可能已损坏")

        # 根据文件后缀名判断框架并加载
        supported_suffixes = ['.pt', '.pth', '.bin', '.ckpt']
        
        if model_path.suffix in supported_suffixes:
            try:
                print(f"🔍 使用 PyTorch 加载模型...")
                model = torch.load(model_path, map_location='cpu')
                print(f"✅ 模型加载成功")
                
                if hasattr(model, 'eval'):
                    model.eval()  # 设置为评估模式
                    print("✅ 模型设置为评估模式")
                    
                return model
            except Exception as e:
                raise RuntimeError(f"加载模型失败: {e}")
        else:
            raise ValueError(f"不支持的模型格式: {model_path.suffix}。支持的格式: {supported_suffixes}")