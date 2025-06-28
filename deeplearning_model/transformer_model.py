import torch.nn as nn
from report_ai_system.deeplearning_model.positional_encoding import PositionalEncoding

class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dropout=0.1):
        """
        input_dim: 입력 특성수 (여기서는 2: BeatPosition, Switch)
        d_model: Transformer의 내부 차원
        nhead: Multi-head Attention의 head 수
        num_layers: Transformer Encoder 레이어 수
        """
        super(TransformerModel, self).__init__()
        
        # 입력 차원을 d_model으로 올리는 선형 변환; 후에 Positional Encoding 적용
        self.input_fc = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model*4,
            dropout=dropout, 
            activation="relu",
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 마지막 시퀀스 출력에서 로드 카운트를 예측 (회귀 문제)
        self.fc_out = nn.Linear(d_model, 3)  
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

        
    def forward(self, x):
        """
        x: (batch, seq_length, input_dim)
        """
        # 선형 변환으로 차원 확장
        x = self.input_fc(x)              # (batch, seq_length, d_model)
        x = self.pos_encoder(x)           # positional encoding 적용
        

        x = self.transformer(x)           # (batch, seq_length, d_model)
        
        # 마지막 타임스탭의 출력 선택하여 Fully Connected layer 통과: (batch, d_model) -> (batch, 1)
        out = self.fc_out(x[:, 0, :])
        # 예측 값은 회귀 값이므로 학습과정에서는 그대로 두고, 추론(혹은 후처리) 시 반올림 처리 가능
        return self.softmax(out)