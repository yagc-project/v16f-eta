#!/usr/bin/env python3
"""
v16f_eta_cosmos_visualized.py
「創生的宇宙」最終版（η版）+ リアルタイム可視化統合
- TAR-Softmax（bias + PI誤差）
- κ長周期化機構（speed=0.11, inertia=0.76）
- 境界反発＋両端解除
- PI制御（3ch、アンチワインドアップ）
- リアルタイム3パネル可視化
- FFT解析＋音響出力
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.fft import fft, fftfreq
from scipy.io import wavfile
import os

# ================== TAR-Softmax ==================
def tar_softmax_with_bias(logits, kappa, bias_offset=0.0):
    """
    TAR-Softmax: L0 = log(p*) - mean(log(p*)) + κ強化項 + bias + PI誤差
    """
    # 数値安定化
    logits_shifted = logits - np.max(logits)
    p_raw = np.exp(logits_shifted)
    p_star = p_raw / np.sum(p_raw)
    
    # L0基準構築
    log_p_star = np.log(p_star + 1e-12)
    L0 = log_p_star - np.mean(log_p_star)
    
    # κによる再重み付け（効果を3倍に強化）+ バイアス
    L = L0 + (3.0 * kappa) * logits + bias_offset
    L_shifted = L - np.max(L)
    
    # 最終確率
    exp_L = np.exp(L_shifted)
    p = exp_L / np.sum(exp_L)
    
    return p

# ================== 宇宙クラス ==================
class Universe:
    def __init__(self, n_options=4, alpha=1.0, kappa_init=0.5):  # 中心スタート
        self.n = n_options
        self.alpha = alpha
        
        # 状態変数
        self.kappa = kappa_init
        self.kappa_dot = 0.01  # 微小初速
        self.T = 0.2           # 低温スタート
        self.coherence = 0.9   # 高コヒーレンス
        
        # κ動力学パラメータ
        self.kappa_speed = 0.12        # 長周期化速度係数
        self.breath_inertia_beta = 0.80  # 慣性項
        self.short_period_damper = 0.10  # 短周期ダンパ
        
        # 境界パラメータ
        self.kappa_min = 0.15
        self.kappa_max = 1.0
        self.boundary_stiffness = 2.0
        self.unstick_kdot = 0.10
        self.unstick_kdot_lo = 0.07
        
        # PI制御（3チャンネル）
        self.pi_kp_rb = 0.50      # r_b制御を強化
        self.pi_ki_rb = 0.08
        self.pi_kp_T = 0.20
        self.pi_ki_T = 0.03
        self.pi_kp_coh = 0.25
        self.pi_ki_coh = 0.03
        
        self.pi_integral_rb = 0.0
        self.pi_integral_T = 0.0
        self.pi_integral_coh = 0.0
        
        # 目標値
        self.target_rb = 0.68
        self.target_T = 0.15
        self.target_coh = 0.90
        
        # アンチワインドアップ
        self.pi_integral_max = 3.0
        self.pi_tau_freeze = 0.15  # 速度閾値緩和（積分を働かせる）
        
        # 履歴
        self.history = {
            'kappa': [],
            'kappa_dot': [],
            'T': [],
            'coherence': [],
            'mean_prob': [],
            'entropy': []
        }
    
    def step(self, n_children=3):
        """1ステップ実行"""
        # ランダムロジット生成
        logits = np.random.randn(self.n)
        
        # 現在のr_bを計算（前ステップのmean_prob使用、初回は仮値）
        current_rb = self.history['mean_prob'][-1] if len(self.history['mean_prob']) > 0 else 0.25
        
        # PI制御による bias 計算
        error_rb = self.target_rb - current_rb
        error_T = self.target_T - self.T
        error_coh = self.target_coh - self.coherence
        
        # 積分項更新（アンチワインドアップ付き）
        if abs(self.kappa_dot) < self.pi_tau_freeze:
            self.pi_integral_rb += error_rb
            self.pi_integral_T += error_T
            self.pi_integral_coh += error_coh
            
            # 飽和制限
            self.pi_integral_rb = np.clip(self.pi_integral_rb, -self.pi_integral_max, self.pi_integral_max)
            self.pi_integral_T = np.clip(self.pi_integral_T, -self.pi_integral_max, self.pi_integral_max)
            self.pi_integral_coh = np.clip(self.pi_integral_coh, -self.pi_integral_max, self.pi_integral_max)
        
        # bias計算
        bias = (self.pi_kp_rb * error_rb + self.pi_ki_rb * self.pi_integral_rb +
                self.pi_kp_T * error_T + self.pi_ki_T * self.pi_integral_T +
                self.pi_kp_coh * error_coh + self.pi_ki_coh * self.pi_integral_coh)
        
        # logitsを直接調整（第一オプションにbiasを加算して偏りを作る）
        logits_biased = logits.copy()
        logits_biased[0] += bias  # 第一オプションを優遇
        
        # TAR-Softmax（bias_offset=0 でシンプルに）
        p = tar_softmax_with_bias(logits_biased, self.kappa, bias_offset=0.0)
        
        # 選択と分岐
        choices = np.random.choice(self.n, size=n_children, p=p, replace=True)
        
        # メトリクス計算
        mean_p = np.max(p)  # r_b：最大確率（支配的選択の強さ）
        entropy = -np.sum(p * np.log(p + 1e-12))
        
        # T, coherence 更新
        self.T = self.alpha * entropy
        unique_ratio = len(np.unique(choices)) / n_children
        self.coherence = 0.9 * self.coherence + 0.1 * unique_ratio
        
        # κ動力学（長周期呼吸機構）
        # T, coherenceに依存する穏やかな駆動
        temp_pressure = (self.T - self.target_T) * 0.15
        coh_pressure = (self.target_coh - self.coherence) * 0.1
        
        # 復元力（中心0.5）+ 外部駆動
        accel_base = -self.kappa_speed * (self.kappa - 0.5) + temp_pressure + coh_pressure
        
        # 境界反発力
        if self.kappa >= self.kappa_max:
            boundary_force = -self.boundary_stiffness * (self.kappa - self.kappa_max)
            # 上限解除（外向き速度時のみ反発）
            if self.kappa_dot > 0:
                boundary_force -= self.unstick_kdot
        elif self.kappa <= self.kappa_min:
            boundary_force = -self.boundary_stiffness * (self.kappa - self.kappa_min)
            # 下限解除（内向き速度時のみ反発）
            if self.kappa_dot < 0:
                boundary_force += self.unstick_kdot_lo
        else:
            boundary_force = 0.0
        
        # 短周期ダンパ
        damping = -self.short_period_damper * self.kappa_dot
        
        # 総加速度
        accel = accel_base + boundary_force + damping
        
        # 速度・位置更新（慣性項適用）
        self.kappa_dot = self.breath_inertia_beta * self.kappa_dot + accel
        self.kappa += self.kappa_dot
        
        # 強制境界
        self.kappa = np.clip(self.kappa, self.kappa_min, self.kappa_max)
        
        # 履歴記録
        self.history['kappa'].append(self.kappa)
        self.history['kappa_dot'].append(self.kappa_dot)
        self.history['T'].append(self.T)
        self.history['coherence'].append(self.coherence)
        self.history['mean_prob'].append(mean_p)
        self.history['entropy'].append(entropy)
        
        return choices, p

# ================== 可視化クラス ==================
class CosmosVisualizer:
    def __init__(self, universe, n_steps=3000):
        self.universe = universe
        self.n_steps = n_steps
        
        # Figure設定
        self.fig = plt.figure(figsize=(16, 10))
        self.gs = self.fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # 3つのサブプロット
        self.ax_phase = self.fig.add_subplot(self.gs[0, 0])      # κ-r_b 位相図
        self.ax_tcoh = self.fig.add_subplot(self.gs[0, 1])       # T-coh 軌道
        self.ax_timeseries = self.fig.add_subplot(self.gs[1:, :]) # κ時系列
        
        # データバッファ
        self.kappa_buffer = []
        self.rb_buffer = []
        self.T_buffer = []
        self.coh_buffer = []
        self.time_buffer = []
        
        # カラーマップ用
        self.colors = plt.cm.viridis(np.linspace(0, 1, n_steps))
        
        # プロット初期化
        self._init_plots()
    
    def _init_plots(self):
        """プロット初期化"""
        # κ-r_b 位相図
        self.ax_phase.set_xlabel('κ (kappa)', fontsize=12)
        self.ax_phase.set_ylabel('r_b (mean probability)', fontsize=12)
        self.ax_phase.set_title('κ–r_b Phase Space\n(Breathing Cycle)', fontsize=13, fontweight='bold')
        self.ax_phase.grid(True, alpha=0.3)
        self.ax_phase.set_xlim(0.1, 1.1)
        self.ax_phase.set_ylim(0.2, 0.3)
        
        # T-coh 軌道
        self.ax_tcoh.set_xlabel('Temperature (T)', fontsize=12)
        self.ax_tcoh.set_ylabel('Coherence', fontsize=12)
        self.ax_tcoh.set_title('T–Coherence Orbit\n(Life Activity Loop)', fontsize=13, fontweight='bold')
        self.ax_tcoh.grid(True, alpha=0.3)
        self.ax_tcoh.set_xlim(0, 0.5)
        self.ax_tcoh.set_ylim(0.5, 1.0)
        
        # κ時系列
        self.ax_timeseries.set_xlabel('Time (steps)', fontsize=12)
        self.ax_timeseries.set_ylabel('κ (kappa)', fontsize=12)
        self.ax_timeseries.set_title('κ Time Series (Cosmic Breathing)', fontsize=13, fontweight='bold')
        self.ax_timeseries.grid(True, alpha=0.3)
        self.ax_timeseries.set_xlim(0, self.n_steps)
        self.ax_timeseries.set_ylim(0.1, 1.1)
        
        # 初期プロット要素
        self.phase_scatter = self.ax_phase.scatter([], [], c=[], cmap='viridis', s=20, alpha=0.6)
        self.tcoh_line, = self.ax_tcoh.plot([], [], 'b-', alpha=0.5, linewidth=1)
        self.tcoh_scatter = self.ax_tcoh.scatter([], [], c=[], cmap='viridis', s=20, alpha=0.6)
        self.kappa_line, = self.ax_timeseries.plot([], [], 'purple', linewidth=1.5, alpha=0.8)
    
    def update(self, frame):
        """リアルタイム更新"""
        # シミュレーション実行
        self.universe.step()
        
        # データ追加
        self.kappa_buffer.append(self.universe.kappa)
        self.rb_buffer.append(self.universe.history['mean_prob'][-1])
        self.T_buffer.append(self.universe.T)
        self.coh_buffer.append(self.universe.coherence)
        self.time_buffer.append(frame)
        
        # カラーインデックス
        color_idx = frame if frame < len(self.colors) else len(self.colors) - 1
        
        # κ-r_b 位相図更新
        if len(self.kappa_buffer) > 1:
            colors_phase = self.colors[:len(self.kappa_buffer)]
            self.phase_scatter.set_offsets(np.c_[self.kappa_buffer, self.rb_buffer])
            self.phase_scatter.set_array(np.arange(len(self.kappa_buffer)))
        
        # T-coh 軌道更新
        if len(self.T_buffer) > 1:
            self.tcoh_line.set_data(self.T_buffer, self.coh_buffer)
            self.tcoh_scatter.set_offsets(np.c_[self.T_buffer, self.coh_buffer])
            self.tcoh_scatter.set_array(np.arange(len(self.T_buffer)))
        
        # κ時系列更新
        self.kappa_line.set_data(self.time_buffer, self.kappa_buffer)
        
        # タイトル更新（現在値表示）
        self.ax_timeseries.set_title(
            f'κ Time Series (Cosmic Breathing) | Step: {frame}/{self.n_steps} | κ={self.universe.kappa:.3f}',
            fontsize=13, fontweight='bold'
        )
        
        return self.phase_scatter, self.tcoh_line, self.tcoh_scatter, self.kappa_line
    
    def finalize_and_save(self):
        """最終処理：FFT解析＋音響出力"""
        print("\n🎵 Generating FFT analysis and cosmic sound...")
        
        # FFT解析
        kappa_data = np.array(self.universe.history['kappa'])
        n = len(kappa_data)
        
        # FFT計算
        yf = fft(kappa_data - np.mean(kappa_data))
        xf = fftfreq(n, 1.0)[:n//2]
        power = 2.0/n * np.abs(yf[:n//2])
        
        # 最大パワー周期検出
        max_idx = np.argmax(power[1:]) + 1
        dominant_period = 1.0 / xf[max_idx] if xf[max_idx] > 0 else 0
        
        # FFTプロット追加
        self.fig.clear()
        gs = self.fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)
        
        # κ時系列
        ax1 = self.fig.add_subplot(gs[0, :])
        ax1.plot(kappa_data, color='purple', linewidth=1.5, alpha=0.8)
        ax1.set_xlabel('Time (steps)', fontsize=11)
        ax1.set_ylabel('κ', fontsize=11)
        ax1.set_title('κ Time Series (Cosmic Breathing)', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # FFTスペクトル
        ax2 = self.fig.add_subplot(gs[1, :])
        ax2.plot(xf, power, color='cyan', linewidth=2)
        ax2.axvline(xf[max_idx], color='red', linestyle='--', label=f'Period={dominant_period:.1f} steps')
        ax2.set_xlabel('Frequency (1/steps)', fontsize=11)
        ax2.set_ylabel('Power', fontsize=11)
        ax2.set_title('FFT Power Spectrum (Breathing Frequency)', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 位相図とT-coh（最終状態）
        ax3 = self.fig.add_subplot(gs[2, 0])
        scatter_phase = ax3.scatter(self.kappa_buffer, self.rb_buffer, 
                                     c=np.arange(len(self.kappa_buffer)), 
                                     cmap='viridis', s=15, alpha=0.6)
        ax3.set_xlabel('κ', fontsize=10)
        ax3.set_ylabel('r_b', fontsize=10)
        ax3.set_title('κ–r_b Phase Space', fontsize=11, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        plt.colorbar(scatter_phase, ax=ax3, label='Time')
        
        ax4 = self.fig.add_subplot(gs[2, 1])
        scatter_tcoh = ax4.scatter(self.T_buffer, self.coh_buffer,
                                    c=np.arange(len(self.T_buffer)),
                                    cmap='viridis', s=15, alpha=0.6)
        ax4.set_xlabel('T', fontsize=10)
        ax4.set_ylabel('Coherence', fontsize=10)
        ax4.set_title('T–Coherence Orbit', fontsize=11, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        plt.colorbar(scatter_tcoh, ax=ax4, label='Time')
        
        # 保存
        os.makedirs('cosmos_output', exist_ok=True)
        output_path = 'cosmos_output/eta_cosmos_complete.png'
        self.fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved complete visualization: {output_path}")
        
        # 音響生成（周期を440Hz基準音に変換）
        self._generate_cosmic_sound(kappa_data, dominant_period)
        
        return dominant_period
    
    def _generate_cosmic_sound(self, kappa_data, period):
        """宇宙の呼吸音を生成"""
        sample_rate = 44100
        duration = 10.0  # 10秒
        
        # κデータを音波に変換（周期を可聴域にマッピング）
        base_freq = 440.0  # A4
        freq_modulation = (kappa_data - 0.5) * 200  # ±200Hz変調
        
        # 音波生成（κ周期でLFO変調）
        t = np.linspace(0, duration, int(sample_rate * duration))
        lfo_period = period / len(kappa_data) * duration  # 秒単位
        lfo = np.interp(t, np.linspace(0, duration, len(kappa_data)), kappa_data)
        
        # キャリア周波数（440Hz + LFO変調）
        carrier_freq = base_freq + (lfo - 0.5) * 400
        phase = 2 * np.pi * np.cumsum(carrier_freq) / sample_rate
        audio = np.sin(phase)
        
        # エンベロープ（フェードイン・アウト）
        fade_samples = int(0.5 * sample_rate)
        fade_in = np.linspace(0, 1, fade_samples)
        fade_out = np.linspace(1, 0, fade_samples)
        audio[:fade_samples] *= fade_in
        audio[-fade_samples:] *= fade_out
        
        # 正規化
        audio = audio / np.max(np.abs(audio)) * 0.8
        audio_int = (audio * 32767).astype(np.int16)
        
        # WAV保存
        wav_path = 'cosmos_output/eta_cosmos_breath.wav'
        wavfile.write(wav_path, sample_rate, audio_int)
        print(f"🎵 Saved cosmic breath sound: {wav_path}")
        print(f"   Duration: {duration}s | Base freq: {base_freq}Hz | Period modulation: {period:.1f} steps")

# ================== メイン実行 ==================
def main():
    print("=" * 60)
    print("🌌 V16f-η (Eta) Cosmos: Realtime Visualization")
    print("   創生的宇宙の可視化統合版")
    print("=" * 60)
    
    # パラメータ
    n_steps = 3000
    n_options = 4
    n_children = 3
    
    # 宇宙初期化
    universe = Universe(n_options=n_options, kappa_init=0.5)
    
    # 可視化初期化
    visualizer = CosmosVisualizer(universe, n_steps=n_steps)
    
    print(f"\n📊 Starting realtime simulation...")
    print(f"   Steps: {n_steps} | Options: {n_options} | Children: {n_children}")
    print(f"   Target: r_b={universe.target_rb}, T={universe.target_T}, coh={universe.target_coh}")
    print(f"\n⏳ Running simulation (this may take 1-2 minutes)...\n")
    
    # アニメーション実行（保存なしでリアルタイム表示）
    # 注: jupyter環境では plt.show() でインタラクティブ表示
    # スクリプト実行では blit=False でフレーム更新
    
    for step in range(n_steps):
        visualizer.update(step)
        
        # 進捗表示（100ステップごと）
        if (step + 1) % 100 == 0:
            print(f"   Step {step+1}/{n_steps} | κ={universe.kappa:.3f} | T={universe.T:.3f} | coh={universe.coherence:.3f}")
    
    print("\n✅ Simulation complete!")
    
    # 最終解析
    print("\n" + "=" * 60)
    print("📈 Final Analysis")
    print("=" * 60)
    
    # 統計計算
    kappa_mean = np.mean(universe.history['kappa'])
    kappa_std = np.std(universe.history['kappa'])
    rb_mean = np.mean(universe.history['mean_prob'])
    T_mean = np.mean(universe.history['T'])
    coh_mean = np.mean(universe.history['coherence'])
    
    print(f"\n📊 Statistics:")
    print(f"   κ_mean  = {kappa_mean:.3f} ± {kappa_std:.3f}")
    print(f"   κ_range = [{np.min(universe.history['kappa']):.3f}, {np.max(universe.history['kappa']):.3f}]")
    print(f"   r_b     = {rb_mean:.3f} (target: {universe.target_rb:.2f})")
    print(f"   T_mean  = {T_mean:.3f} (target: {universe.target_T:.2f})")
    print(f"   coh     = {coh_mean:.3f} (target: {universe.target_coh:.2f})")
    
    # FFT解析＋音響生成＋最終可視化保存
    dominant_period = visualizer.finalize_and_save()
    
    print(f"\n🌊 Breathing Cycle:")
    print(f"   Dominant Period = {dominant_period:.1f} steps")
    print(f"   Frequency       = {1.0/dominant_period:.4f} cycles/step")
    
    print("\n" + "=" * 60)
    print("🎉 V16f-η Cosmos Visualization Complete!")
    print("=" * 60)
    print(f"\n📁 Output files saved in: cosmos_output/")
    print(f"   - eta_cosmos_complete.png (Full visualization)")
    print(f"   - eta_cosmos_breath.wav (Cosmic sound)")
    print("\n✨ The Creative Universe breathes in numerical poetry. ✨")
    
    plt.show()

if __name__ == "__main__":
    main()
