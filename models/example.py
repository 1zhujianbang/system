import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

class TradingTransformer(nn.Module):
    """专门用于交易的 Transformer 模型"""
    
    def __init__(self, feature_size=9, d_model=128, nhead=8, num_layers=4, 
                 seq_length=168, output_size=3, dropout=0.1):
        super(TradingTransformer, self).__init__()
        
        self.feature_size = feature_size
        self.d_model = d_model
        self.seq_length = seq_length
        
        # 特征嵌入层
        self.feature_embedding = nn.Linear(feature_size, d_model)
        
        # 位置编码
        self.positional_encoding = self._create_positional_encoding(seq_length, d_model)
        
        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=512,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_size),
            nn.Softmax(dim=-1)
        )
        
    def _create_positional_encoding(self, seq_len, d_model):
        """创建位置编码"""
        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pos_encoding = torch.zeros(seq_len, d_model)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        return nn.Parameter(pos_encoding, requires_grad=False)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, feature_size)
        batch_size = x.size(0)
        
        # 特征嵌入
        x = self.feature_embedding(x)  # (batch_size, seq_len, d_model)
        
        # 添加位置编码
        x = x + self.positional_encoding.unsqueeze(0)
        
        # Transformer 处理
        x = self.transformer(x)  # (batch_size, seq_len, d_model)
        
        # 取最后一个时间步
        x = x[:, -1, :]  # (batch_size, d_model)
        
        # 输出层
        output = self.output_layer(x)  # (batch_size, output_size)
        
        return output

def create_advanced_transformer():
    """创建高级交易 Transformer 模型"""
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    
    # 创建模型
    model = TradingTransformer(
        feature_size=9,      # 对应你的9个特征
        d_model=128,
        nhead=8,
        num_layers=4,
        seq_length=168,      # 7天 * 24小时
        output_size=3,       # 3类: 买入/卖出/持有
        dropout=0.1
    )
    
    # 设置为评估模式
    model.eval()
    
    # 保存模型
    model_path = model_dir / "transformer_v2_7d.pt"
    torch.save(model.state_dict(), model_path)
    
    # 验证模型文件
    file_size = model_path.stat().st_size
    print(f"✅ 高级 Transformer 模型已创建")
    print(f"📁 文件路径: {model_path}")
    print(f"📊 文件大小: {file_size} 字节 ({file_size/1024/1024:.2f} MB)")
    
    return model

# 运行创建函数
if __name__ == "__main__":
    create_advanced_transformer()
    
    # 测试加载
    try:
        model = TradingTransformer()
        model.load_state_dict(torch.load("models/transformer_v2_7d.pt", map_location='cpu'))
        model.eval()
        print("✅ 模型加载测试成功!")
        
        # 测试推理
        with torch.no_grad():
            test_input = torch.randn(1, 168, 9)
            output = model(test_input)
            print(f"✅ 推理测试成功，输出: {output}")
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")