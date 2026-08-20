import math
import matplotlib.pyplot as plt

# ==========================================
# 1. Tek Nöron Forward Pass
# ==========================================
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def single_neuron_forward(x, w, b):
    # z = w1*x1 + w2*x2 + ... + b
    z = sum(xi * wi for xi, wi in zip(x, w)) + b
    return sigmoid(z)

# Örnek tek nöron testi (2 girişli)
sample_x = [2.0, 3.0]
sample_w = [0.5, -1.0]
sample_b = 0.1
output_single = single_neuron_forward(sample_x, sample_w, sample_b)
print(f"1. Tek Nöron Çıktısı: {output_single:.4f}")


# ==========================================
# 2. Çok Nöronlu Katman (Layer) Forward Pass
# ==========================================
def layer_forward(inputs, weights_matrix, biases):
    # weights_matrix: her nöron için ağırlık listesi
    layer_outputs = []
    for w_neuron, b_neuron in zip(weights_matrix, biases):
        out = single_neuron_forward(inputs, w_neuron, b_neuron)
        layer_outputs.append(out)
    return layer_outputs

# Örnek: 2 giriş alan, 3 nörondan oluşan bir katman
layer_weights = [
    [0.2, 0.8],   # 1. nöron ağırlıkları
    [0.5, -0.4],  # 2. nöron ağırlıkları
    [-0.1, 0.9]   # 3. nöron ağırlıkları
]
layer_biases = [0.1, -0.2, 0.0]
output_layer = layer_forward(sample_x, layer_weights, layer_biases)
print(f"2. Katman Çıktısı (3 nöron): {[round(o, 4) for o in output_layer]}")


# ==========================================
# 3. Basit Loss Fonksiyonu (Mean Squared Error)
# ==========================================
def calculate_loss(y_pred, y_true):
    # Tek örnek için MSE: (y_pred - y_true)^2
    return (y_pred - y_true) ** 2


# ==========================================
# 4. Parametreleri Manuel Değiştirme ve Loss Eğrisi
# ==========================================
target_y = 0.8
test_x = [1.5]
fixed_b = 0.0

w_values = [i * 0.1 for i in range(-30, 31)]  # -3.0 ile +3.0 arası ağırlıklar
losses = []

for w in w_values:
    pred = single_neuron_forward(test_x, [w], fixed_b)
    l = calculate_loss(pred, target_y)
    losses.append(l)

# Loss eğrisini çizdirme
plt.figure(figsize=(6, 4))
plt.plot(w_values, losses, label="Loss (MSE)")
plt.xlabel("Ağırlık (w)")
plt.ylabel("Loss")
plt.title("Parametre Değişimine Göre Loss Eğrisi")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("loss_curve.png")
plt.show()


# ==========================================
# 5. Sayısal Türev ile Gradient Descent Döngüsü
# ==========================================
# Hedef: Tek girişli nöronu target_y değerine yaklaştırmak
x_val = [2.0]
y_true = 0.85

# Başlangıç parametreleri
w = 0.1
b = 0.0
lr = 0.5       # Learning rate
h = 0.0001     # Sayısal türev için küçük epsilon (Karpathy stili)
epochs = 50

print("\n--- Gradient Descent Başlıyor ---")
for epoch in range(epochs):
    # Mevcut loss
    pred = single_neuron_forward(x_val, [w], b)
    loss = calculate_loss(pred, y_true)
    
    # w için sayısal türev: dLoss / dw = (Loss(w + h) - Loss(w)) / h
    pred_w_plus_h = single_neuron_forward(x_val, [w + h], b)
    loss_w_plus_h = calculate_loss(pred_w_plus_h, y_true)
    grad_w = (loss_w_plus_h - loss) / h
    
    # b için sayısal türev: dLoss / db = (Loss(b + h) - Loss(b)) / h
    pred_b_plus_h = single_neuron_forward(x_val, [w], b + h)
    loss_b_plus_h = calculate_loss(pred_b_plus_h, y_true)
    grad_b = (loss_b_plus_h - loss) / h
    
    # Parametre güncelleme
    w -= lr * grad_w
    b -= lr * grad_b
    
    if epoch % 10 == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch:02d} | Loss: {loss:.6f} | Pred: {pred:.4f} (Target: {y_true}) | w: {w:.4f}, b: {b:.4f}")