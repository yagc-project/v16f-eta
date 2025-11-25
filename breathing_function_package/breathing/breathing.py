"""
Breathing Function Package
==========================
A universal activation function inspired by YAGC cosmology (V16-V28).

The breathing function models natural rhythmic patterns that emerge in
complex systems - from biological respiration to information processing.

Core Concept:
    α(I) = α₀ tanh^n(I/I_c)
    
Where:
    - I: input intensity (information density)
    - I_c: critical threshold
    - α₀: maximum amplitude
    - n: sharpness parameter

Author: YAGC Project (Yoshida, ChatGPT A., Claude C., Gemini G.)
License: MIT
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Callable, List
from enum import Enum
import time


class BreathPhase(Enum):
    """呼吸の位相"""
    INHALE = "inhale"       # 吸気（立ち上がり）
    HOLD = "hold"           # 保持
    EXHALE = "exhale"       # 呼気（減衰）
    REST = "rest"           # 休息（スタンバイ）


@dataclass
class BreathConfig:
    """呼吸関数の設定"""
    # 基本パラメータ
    alpha_0: float = 1.0        # 最大振幅（深さの上限）
    I_c: float = 1.0            # 臨界閾値
    n: int = 1                  # シャープネス（1=緩やか, 3=急峻）
    
    # 動的パラメータ
    min_level: float = 0.1      # 最低活性レベル（スタンバイ）
    warmup_rate: float = 0.5    # 立ち上がり速度
    decay_rate: float = 0.1     # 減衰速度（一気に下げない）
    
    # 時間制御
    max_cycles: Optional[int] = None    # 回数上限（Noneで無制限）
    max_duration: Optional[float] = None # 時間上限（秒）
    reset_interval: Optional[float] = None  # リセット間隔（秒）
    
    # 日内変動
    diurnal_enabled: bool = False   # 日内変動を有効化
    day_depth: float = 1.0          # 昼間の深さ係数
    night_depth: float = 0.3        # 夜間の深さ係数
    day_start: int = 6              # 昼開始時刻（時）
    night_start: int = 22           # 夜開始時刻（時）


@dataclass
class BreathState:
    """呼吸関数の内部状態"""
    current_value: float = 0.1      # 現在の呼吸値
    target_value: float = 0.1       # 目標値
    intensity: float = 0.0          # 入力強度
    phase: BreathPhase = BreathPhase.REST
    cycle_count: int = 0            # 呼吸サイクル数
    start_time: float = field(default_factory=time.time)
    last_reset: float = field(default_factory=time.time)
    last_update: float = field(default_factory=time.time)
    history: List[float] = field(default_factory=list)


class BreathingFunction:
    """
    呼吸関数クラス
    
    Usage:
        breath = BreathingFunction()
        
        # 基本的な使い方
        value = breath.compute(input_intensity=0.5)
        
        # 動的更新（UIなど）
        breath.update(mouse_velocity)
        current = breath.get_value()
        
        # 日内変動
        breath = BreathingFunction(BreathConfig(diurnal_enabled=True))
    """
    
    def __init__(self, config: Optional[BreathConfig] = None):
        self.config = config or BreathConfig()
        self.state = BreathState()
        self._callbacks: List[Callable] = []
    
    # =========================================================================
    # Core Function: 呼吸関数の計算
    # =========================================================================
    
    def compute(self, I: float) -> float:
        """
        呼吸関数の基本計算
        
        α(I) = α₀ tanh^n(I/I_c)
        
        Args:
            I: 入力強度（情報密度）
            
        Returns:
            活性化値 [0, α₀]
        """
        x = I / self.config.I_c
        base = np.tanh(x)
        result = self.config.alpha_0 * (base ** self.config.n)
        
        # 日内変動を適用
        if self.config.diurnal_enabled:
            result *= self._get_diurnal_factor()
        
        return max(self.config.min_level, result)
    
    def compute_derivative(self, I: float) -> float:
        """
        呼吸関数の導関数
        
        dα/dI = (α₀ n / I_c) tanh^(n-1)(I/I_c) sech²(I/I_c)
        """
        x = I / self.config.I_c
        tanh_x = np.tanh(x)
        sech2_x = 1 - tanh_x**2
        
        if self.config.n == 1:
            return (self.config.alpha_0 / self.config.I_c) * sech2_x
        else:
            return (self.config.alpha_0 * self.config.n / self.config.I_c) * \
                   (tanh_x ** (self.config.n - 1)) * sech2_x
    
    # =========================================================================
    # Dynamic Update: 動的更新システム
    # =========================================================================
    
    def update(self, input_intensity: float, dt: Optional[float] = None) -> float:
        """
        動的更新（UI/ゲーム向け）
        
        - 入力が激しいほどポンプアップ
        - 落ち着いてきたら徐々に下げる（一気には下げない）
        - スタンバイレベルは維持
        
        Args:
            input_intensity: 入力強度（マウス速度、キー入力頻度など）
            dt: 時間刻み（Noneで自動計算）
            
        Returns:
            現在の呼吸値
        """
        now = time.time()
        if dt is None:
            dt = now - self.state.last_update
            dt = min(dt, 0.1)  # 最大100ms
        self.state.last_update = now
        
        # 停止条件のチェック
        if self._check_stop_conditions():
            return self.state.current_value
        
        # リセット条件のチェック
        if self._check_reset_conditions():
            self._reset()
        
        # 目標値を計算
        self.state.intensity = input_intensity
        self.state.target_value = self.compute(input_intensity)
        
        # 現在値を目標に向けて更新（非対称レート）
        current = self.state.current_value
        target = self.state.target_value
        
        if target > current:
            # 上昇時は速く（warmup_rate）
            rate = self.config.warmup_rate
            self.state.phase = BreathPhase.INHALE
        else:
            # 下降時は遅く（decay_rate）- 一気に下げない
            rate = self.config.decay_rate
            self.state.phase = BreathPhase.EXHALE
        
        # 指数的追従
        self.state.current_value = current + (target - current) * rate * dt * 10
        
        # 最低レベルを保証（スタンバイ）
        self.state.current_value = max(
            self.config.min_level, 
            self.state.current_value
        )
        
        # 履歴に追加
        self.state.history.append(self.state.current_value)
        if len(self.state.history) > 1000:
            self.state.history = self.state.history[-500:]
        
        # コールバック呼び出し
        for callback in self._callbacks:
            callback(self.state)
        
        return self.state.current_value
    
    def get_value(self) -> float:
        """現在の呼吸値を取得"""
        return self.state.current_value
    
    def get_state(self) -> BreathState:
        """完全な状態を取得"""
        return self.state
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def _get_diurnal_factor(self) -> float:
        """日内変動係数を計算"""
        hour = time.localtime().tm_hour
        
        if self.config.day_start <= hour < self.config.night_start:
            # 昼間
            return self.config.day_depth
        else:
            # 夜間
            return self.config.night_depth
    
    def _check_stop_conditions(self) -> bool:
        """停止条件をチェック"""
        # 回数上限
        if self.config.max_cycles is not None:
            if self.state.cycle_count >= self.config.max_cycles:
                self.state.phase = BreathPhase.REST
                return True
        
        # 時間上限
        if self.config.max_duration is not None:
            elapsed = time.time() - self.state.start_time
            if elapsed >= self.config.max_duration:
                self.state.phase = BreathPhase.REST
                return True
        
        return False
    
    def _check_reset_conditions(self) -> bool:
        """リセット条件をチェック"""
        if self.config.reset_interval is not None:
            elapsed = time.time() - self.state.last_reset
            if elapsed >= self.config.reset_interval:
                return True
        return False
    
    def _reset(self):
        """状態をリセット"""
        self.state.cycle_count = 0
        self.state.last_reset = time.time()
        self.state.current_value = self.config.min_level
        self.state.history.clear()
    
    def on_update(self, callback: Callable):
        """更新時のコールバックを登録"""
        self._callbacks.append(callback)
    
    # =========================================================================
    # Presets: よく使う設定のプリセット
    # =========================================================================
    
    @classmethod
    def calm(cls) -> 'BreathingFunction':
        """穏やかな呼吸（瞑想、バックグラウンド処理向け）"""
        return cls(BreathConfig(
            alpha_0=0.5,
            n=1,
            warmup_rate=0.2,
            decay_rate=0.05,
            min_level=0.1
        ))
    
    @classmethod
    def active(cls) -> 'BreathingFunction':
        """活発な呼吸（ゲーム、インタラクティブUI向け）"""
        return cls(BreathConfig(
            alpha_0=1.0,
            n=2,
            warmup_rate=0.8,
            decay_rate=0.2,
            min_level=0.15
        ))
    
    @classmethod
    def adaptive(cls) -> 'BreathingFunction':
        """適応的呼吸（日内変動あり）"""
        return cls(BreathConfig(
            alpha_0=1.0,
            n=1,
            warmup_rate=0.5,
            decay_rate=0.1,
            min_level=0.1,
            diurnal_enabled=True,
            day_depth=1.0,
            night_depth=0.3
        ))
    
    @classmethod
    def kernel(cls, kappa: float = 0.55, rb: float = 0.5, tau: float = 0.25) -> 'BreathingFunction':
        """V16カーネルモード（CHNOPS対応）"""
        # V16の3変数からI_cを動的に計算
        I_c = kappa * rb * (1 + np.sin(2 * np.pi * tau))
        return cls(BreathConfig(
            alpha_0=1.0,
            I_c=max(0.1, I_c),
            n=1,
            warmup_rate=0.5,
            decay_rate=0.1
        ))


# =============================================================================
# Convenience Functions
# =============================================================================

def breath(I: float, I_c: float = 1.0, n: int = 1) -> float:
    """
    シンプルな呼吸関数
    
    Args:
        I: 入力強度
        I_c: 臨界閾値
        n: シャープネス
        
    Returns:
        活性化値 [0, 1]
    """
    return np.tanh(I / I_c) ** n


def breath_array(I: np.ndarray, I_c: float = 1.0, n: int = 1) -> np.ndarray:
    """NumPy配列用の呼吸関数"""
    return np.tanh(I / I_c) ** n


# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    print("=" * 60)
    print("Breathing Function Package - Demo")
    print("=" * 60)
    
    # 1. 基本的な呼吸関数のプロット
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    I = np.linspace(0, 3, 100)
    
    # (A) 異なるnでの呼吸関数
    ax = axes[0, 0]
    for n in [1, 2, 3]:
        bf = BreathingFunction(BreathConfig(n=n))
        y = [bf.compute(i) for i in I]
        ax.plot(I, y, label=f'n={n}', linewidth=2)
    ax.axhline(0.2, color='red', linestyle='--', alpha=0.5, label='20% threshold')
    ax.set_xlabel('Input Intensity (I)')
    ax.set_ylabel('Activation α(I)')
    ax.set_title('(A) Breathing Function: Effect of Sharpness n')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # (B) 動的応答シミュレーション
    ax = axes[0, 1]
    bf = BreathingFunction.active()
    
    # 模擬入力：急激な上昇→ゆっくり下降
    t = np.linspace(0, 10, 500)
    input_signal = np.zeros_like(t)
    input_signal[(t > 1) & (t < 3)] = 1.5
    input_signal[(t > 5) & (t < 6)] = 2.0
    input_signal[(t > 7) & (t < 7.5)] = 0.8
    
    output = []
    for i, inp in enumerate(input_signal):
        bf.update(inp, dt=0.02)
        output.append(bf.get_value())
    
    ax.plot(t, input_signal, 'b--', alpha=0.5, label='Input', linewidth=1)
    ax.plot(t, output, 'r-', label='Breathing Response', linewidth=2)
    ax.fill_between(t, 0, output, alpha=0.3, color='red')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Value')
    ax.set_title('(B) Dynamic Response: Fast Rise, Slow Decay')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # (C) 日内変動
    ax = axes[1, 0]
    hours = np.arange(0, 24, 0.5)
    bf_diurnal = BreathingFunction.adaptive()
    
    diurnal_values = []
    for h in hours:
        # 時刻をシミュレート
        factor = bf_diurnal.config.day_depth if 6 <= h < 22 else bf_diurnal.config.night_depth
        diurnal_values.append(factor)
    
    ax.fill_between(hours, 0, diurnal_values, alpha=0.3, color='orange')
    ax.plot(hours, diurnal_values, 'orange', linewidth=2)
    ax.axvline(6, color='gold', linestyle='--', alpha=0.5, label='Day start')
    ax.axvline(22, color='navy', linestyle='--', alpha=0.5, label='Night start')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Depth Factor')
    ax.set_title('(C) Diurnal Variation: Deep by Day, Shallow by Night')
    ax.set_xlim(0, 24)
    ax.legend()
    ax.grid(alpha=0.3)
    
    # (D) プリセット比較
    ax = axes[1, 1]
    presets = {
        'calm': BreathingFunction.calm(),
        'active': BreathingFunction.active(),
    }
    
    for name, bf in presets.items():
        y = [bf.compute(i) for i in I]
        ax.plot(I, y, label=name, linewidth=2)
    
    ax.set_xlabel('Input Intensity (I)')
    ax.set_ylabel('Activation α(I)')
    ax.set_title('(D) Presets: Calm vs Active')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/breathing_function_demo.png', dpi=150, bbox_inches='tight')
    print("\n✅ Demo figure saved!")
    
    # 2. 使用例を表示
    print("\n" + "=" * 60)
    print("Usage Examples")
    print("=" * 60)
    
    print("""
# Basic usage
from breathing import BreathingFunction, breath

# Simple function call
value = breath(0.5)  # → ~0.46

# Create a breathing controller
bf = BreathingFunction()
bf.update(mouse_velocity)  # Update with input
current = bf.get_value()    # Get current value

# Presets
bf_calm = BreathingFunction.calm()      # For meditation apps
bf_active = BreathingFunction.active()  # For games
bf_adaptive = BreathingFunction.adaptive()  # Day/night aware

# Custom configuration
from breathing import BreathConfig
config = BreathConfig(
    alpha_0=1.0,      # Max amplitude
    I_c=1.0,          # Threshold
    n=2,              # Sharpness
    warmup_rate=0.8,  # Fast rise
    decay_rate=0.1,   # Slow decay
    min_level=0.15,   # Standby level
)
bf = BreathingFunction(config)
""")
    
    print("\n✅ Breathing Function Package ready!")
    print("=" * 60)
