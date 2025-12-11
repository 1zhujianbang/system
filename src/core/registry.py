from typing import Callable, Dict, Any, Optional, List, Type
import inspect
from functools import wraps
from pydantic import BaseModel, create_model, ValidationError
from .logging import LoggerManager

class FunctionRegistry:
    _registry: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def register_tool(cls, name: str = None, description: str = None, category: str = "Uncategorized", input_model: Type[BaseModel] = None):
        """
        装饰器：将函数注册到全局功能池
        
        Args:
            name: 自定义工具名称（默认为函数名）
            description: 工具描述（默认为 docstring）
            category: 工具分类
            input_model: 可选的 Pydantic 模型，用于严格的参数验证。如果不提供，尝试根据类型提示自动生成。
        """
        def decorator(func: Callable):
            func_name = name or func.__name__
            doc = description or func.__doc__ or "No description provided."
            
            # 获取参数签名
            try:
                sig = inspect.signature(func)
                parameters = {}
                fields = {}
                
                for k, v in sig.parameters.items():
                    # 忽略 self/cls 参数
                    if k in ['self', 'cls']:
                        continue
                    
                    # 获取类型注解
                    annotation = v.annotation if v.annotation != inspect.Parameter.empty else Any
                    default = v.default if v.default != inspect.Parameter.empty else ...
                    
                    param_info = {
                        "type": str(annotation),
                        "default": str(default) if default != ... else None,
                        "required": default == ...
                    }
                    parameters[k] = param_info
                    fields[k] = (annotation, default)
                
                return_type = str(sig.return_annotation) if sig.return_annotation != inspect.Signature.empty else "Any"
                
                # 如果未提供 input_model，尝试自动生成一个
                model = input_model
                if model is None and fields:
                    try:
                        model = create_model(f"{func_name}Input", **fields)
                    except Exception as e:
                        # 某些复杂类型可能无法自动生成 Model，降级处理
                        LoggerManager.get_logger(__name__).warning(f"Could not auto-generate Pydantic model for {func_name}: {e}")
                        model = None
                        
            except ValueError:
                # 某些内建函数可能无法获取签名
                parameters = {}
                return_type = "Any"
                model = None

            cls._registry[func_name] = {
                "func": func,
                "name": func_name,
                "description": doc.strip(),
                "category": category,
                "parameters": parameters,
                "return_type": return_type,
                "input_model": model, # 存储验证模型
                "module": func.__module__
            }
            
            @wraps(func)
            def wrapper(*args, **kwargs):
                # 运行时验证（可选，如果在 Engine 中统一调用则不必在此验证）
                return func(*args, **kwargs)
            return wrapper
        return decorator

    @classmethod
    def get_tool(cls, name: str) -> Optional[Callable]:
        """通过名称获取工具函数"""
        entry = cls._registry.get(name)
        return entry["func"] if entry else None

    @classmethod
    def get_input_model(cls, name: str) -> Optional[Type[BaseModel]]:
        """获取工具的输入验证模型"""
        entry = cls._registry.get(name)
        return entry.get("input_model") if entry else None

    @classmethod
    def get_metadata(cls, name: str) -> Optional[Dict[str, Any]]:
        """获取工具元数据"""
        entry = cls._registry.get(name)
        if not entry:
            return None
        return {k: v for k, v in entry.items() if k not in ["func", "input_model"]}

    @classmethod
    def get_all_tools(cls) -> Dict[str, Dict[str, Any]]:
        """获取所有已注册工具的元数据（不含函数对象）"""
        return {k: {key: val for key, val in v.items() if key not in ["func", "input_model"]} 
                for k, v in cls._registry.items()}

# 快捷方式
register_tool = FunctionRegistry.register_tool
