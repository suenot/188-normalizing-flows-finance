# Глава 332: Нормализующие потоки для финансов

## Обзор

Нормализующие потоки (Normalizing Flows) - это класс глубоких генеративных моделей, которые обучаются сложным распределениям вероятностей путем преобразования простого базового распределения (например, Гауссовского) через последовательность обратимых, дифференцируемых преобразований. В отличие от других генеративных моделей (VAE, GAN), нормализующие потоки обеспечивают **точное вычисление правдоподобия**, что делает их идеальными для финансовых приложений, где точная оценка плотности критически важна для управления рисками.

## Почему нормализующие потоки для финансов?

### Проблема традиционных подходов

Финансовые доходности заведомо **не являются гауссовскими**:

- **Тяжелые хвосты**: Экстремальные события происходят чаще, чем предсказывает нормальное распределение
- **Асимметрия**: Доходности часто асимметричны (большие падения, чем рост)
- **Изменяющаяся волатильность**: Форма распределения меняется со временем
- **Мультимодальность**: Различные рыночные режимы создают сложные распределения

Традиционные модели риска (VaR, CVaR) предполагают гауссовские доходности, что приводит к:
- Недооценке хвостового риска
- Плохим решениям по хеджированию
- Неожиданным потерям в периоды рыночного стресса

### Решение с нормализующими потоками

Нормализующие потоки изучают **истинное распределение** доходностей:

```
Традиционный подход: Предполагаем X ~ N(μ, σ²) → Недооцениваем хвостовой риск

Нормализующий поток: Изучаем p(X) напрямую → Точная плотность для любой формы
  z ~ N(0, I)     [Простое базовое распределение]
  x = f(z)        [Обратимое преобразование]
  p(x) = p(z)|det(∂f⁻¹/∂x)|  [Точное правдоподобие через замену переменных]
```

## Математические основы

### Формула замены переменных

Ключевой принцип нормализующих потоков - **формула замены переменных**:

Дано:
- Базовое распределение: z ~ p_z(z) (обычно стандартное нормальное)
- Обратимое преобразование: x = f(z), следовательно z = f⁻¹(x)
- Целевое распределение: p_x(x)

Преобразование плотности:

```
p_x(x) = p_z(f⁻¹(x)) |det(J_{f⁻¹}(x))|

где J_{f⁻¹}(x) = ∂f⁻¹(x)/∂x - матрица Якоби
```

Для последовательности K преобразований:

```
z₀ → f₁ → z₁ → f₂ → z₂ → ... → f_K → x

log p(x) = log p(z₀) - Σᵢ log|det(J_{fᵢ})|
```

### Почему важен якобиан

Определитель якобиана учитывает, как преобразование **растягивает или сжимает** пространство:

```
┌────────────────────────────────────────────────────────────┐
│                   ИНТУИЦИЯ ЯКОБИАНА                        │
├────────────────────────────────────────────────────────────┤
│                                                             │
│   Базовое распределение (z)    Целевое распределение (x)  │
│                                                             │
│      ┌─────────┐                   ┌──────────────┐        │
│      │  ***    │      f(z)        │    ***       │        │
│      │ *****   │  ──────────►     │ ***    **    │        │
│      │  ***    │                   │  ****  ***   │        │
│      └─────────┘                   └──────────────┘        │
│                                                             │
│   Гауссово облако             Сложное распределение        │
│   (легко сэмплировать)        (трудно моделировать)        │
│                                                             │
│   Якобиан = Как меняется объем в каждой точке              │
│   |det(J)| > 1 → Пространство расширяется → Плотность ↓   │
│   |det(J)| < 1 → Пространство сжимается → Плотность ↑     │
│                                                             │
└────────────────────────────────────────────────────────────┘
```

## Типы нормализующих потоков

### 1. Аффинные связывающие потоки (RealNVP)

**Ключевая идея**: Разделить вход и применить простые преобразования с вычислимыми якобианами.

```
┌─────────────────────────────────────────────────────────────┐
│                 АФФИННЫЙ СВЯЗЫВАЮЩИЙ СЛОЙ                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Вход: x = [x₁, x₂]  (разделен на две части)              │
│                                                             │
│   Преобразование:                                           │
│     y₁ = x₁                     (без изменений)             │
│     y₂ = x₂ ⊙ exp(s(x₁)) + t(x₁)  (аффинное преобр.)       │
│                                                             │
│   где s() и t() - нейронные сети                           │
│                                                             │
│   Якобиан ТРЕУГОЛЬНЫЙ → det = ∏ exp(s(x₁)) = exp(Σs)      │
│   Очень эффективно! O(D) вместо O(D³)                      │
│                                                             │
│        ┌──────────┐                                         │
│   x₁ ──┤Тождество ├──────────────────────────────► y₁      │
│        └──────────┘                                         │
│             │                                               │
│             ▼                                               │
│        ┌────────┐        ┌─────────────────────┐           │
│   x₂ ──┤Нейросеть├──s,t──►│ y₂ = x₂·exp(s) + t │──► y₂    │
│        └────────┘        └─────────────────────┘           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2. Авторегрессионные потоки (MAF/IAF)

**Masked Autoregressive Flow (MAF)**: Каждое измерение зависит от предыдущих.

```
x₁ = z₁ · σ₁ + μ₁
x₂ = z₂ · σ₂(x₁) + μ₂(x₁)
x₃ = z₃ · σ₃(x₁,x₂) + μ₃(x₁,x₂)
...

Якобиан НИЖНЕТРЕУГОЛЬНЫЙ → det = ∏ σᵢ
```

**Inverse Autoregressive Flow (IAF)**: Обратное направление для быстрой генерации.

```
MAF: Быстрая плотность, медленная генерация
IAF: Быстрая генерация, медленная плотность

┌────────────────────────────────────────┐
│              MAF vs IAF                │
├────────────────────────────────────────┤
│ Операция      │  MAF    │   IAF       │
├───────────────┼─────────┼─────────────┤
│ log p(x)      │ O(1)    │ O(D)        │
│ Генерация     │ O(D)    │ O(1)        │
│ Обучение      │ Быстро  │ Медленно    │
│ Сэмплирование │ Медленно│ Быстро      │
└────────────────────────────────────────┘
```

### 3. Непрерывные нормализующие потоки (Neural ODE)

**Ключевая идея**: Вместо дискретных преобразований определяем непрерывный поток через ОДУ:

```
dz/dt = f(z, t; θ)

Изменение логарифма правдоподобия:
d log p(z)/dt = -tr(∂f/∂z)

Решается численным интегрированием (метод сопряженных)
```

## Архитектура модели

```
┌─────────────────────────────────────────────────────────────────┐
│          НОРМАЛИЗУЮЩИЙ ПОТОК ДЛЯ ФИНАНСОВЫХ ДОХОДНОСТЕЙ        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ВХОДНОЙ СЛОЙ                                                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Данные финансовых доходностей:                            │   │
│  │   - Дневные/часовые доходности                            │   │
│  │   - Мультиактивные доходности (портфель)                  │   │
│  │   - Условные признаки (волатильность, объем и т.д.)       │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              ↓                                   │
│  ПРЕДОБРАБОТКА                                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Стандартизация: (x - μ) / σ                               │   │
│  │ Винсоризация: обрезка экстремальных значений              │   │
│  │ Опционально: добавление контекста для условного потока    │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              ↓                                   │
│  БЛОКИ ПОТОКА (×N)                                              │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ ┌────────────────────────────────────────────────────┐   │   │
│  │ │ Аффинный связывающий слой 1                         │   │   │
│  │ │   x₁ без изменений, x₂ преобр. через s(x₁), t(x₁)  │   │   │
│  │ └────────────────────────────────────────────────────┘   │   │
│  │                         ↓                                │   │
│  │ ┌────────────────────────────────────────────────────┐   │   │
│  │ │ Перестановка / Перемешивание                        │   │   │
│  │ │   Обеспечивает преобразование всех измерений        │   │   │
│  │ └────────────────────────────────────────────────────┘   │   │
│  │                         ↓                                │   │
│  │ ┌────────────────────────────────────────────────────┐   │   │
│  │ │ Аффинный связывающий слой 2                         │   │   │
│  │ │   Противоположное разделение (x₂ без изм., x₁ пр.) │   │   │
│  │ └────────────────────────────────────────────────────┘   │   │
│  │                         ↓                                │   │
│  │ ┌────────────────────────────────────────────────────┐   │   │
│  │ │ Батч-нормализация (опционально)                     │   │   │
│  │ │   Стабилизация обучения                             │   │   │
│  │ └────────────────────────────────────────────────────┘   │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              ↓                                   │
│  БАЗОВОЕ РАСПРЕДЕЛЕНИЕ                                           │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Стандартное гауссово: z ~ N(0, I)                         │   │
│  │ Или t-Стьюдента для тяжелых хвостов: z ~ t(ν, 0, I)       │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              ↓                                   │
│  ВЫХОД                                                           │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ log p(x) = log p(z) + Σ log|det(J_k)|                     │   │
│  │ Сэмплы: z ~ p(z) → x = f(z)                               │   │
│  │ Плотность: x → z = f⁻¹(x) → p(x)                          │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Финансовые приложения

### 1. Оценка плотности для доходностей

```python
def estimate_return_density(model, returns):
    """
    Оценка плотности вероятности доходностей.

    Args:
        model: Обученный нормализующий поток
        returns: Массив значений доходности

    Returns:
        log_prob: Логарифм плотности вероятности в каждой точке
    """
    # Преобразуем доходности в латентное пространство
    z, log_det = model.inverse(returns)

    # Вычисляем лог-вероятность базового распределения
    log_pz = -0.5 * (z**2 + np.log(2*np.pi)).sum(dim=-1)

    # Полная лог-вероятность через замену переменных
    log_prob = log_pz + log_det

    return log_prob
```

### 2. Value at Risk (VaR) с изученными плотностями

Традиционный VaR предполагает гауссовы доходности. С нормализующими потоками:

```
┌─────────────────────────────────────────────────────────────┐
│                    СРАВНЕНИЕ VaR                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Гауссов VaR (недооценивает хвостовой риск):               │
│   VaR_α = μ + σ · Φ⁻¹(α)                                   │
│                                                             │
│   VaR на нормализующем потоке (точный):                     │
│   VaR_α = квантиль из изученного распределения p(x)         │
│   Находится через: ∫_{-∞}^{VaR} p(x)dx = α                 │
│                                                             │
│   Подход Монте-Карло:                                       │
│   1. Сэмплируем N точек из потока: xᵢ ~ p(x)               │
│   2. Сортируем сэмплы                                       │
│   3. VaR_α = x_{⌊αN⌋}                                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3. Conditional Value at Risk (CVaR / Expected Shortfall)

```python
def compute_cvar(model, alpha=0.05, n_samples=100000):
    """
    Вычисление CVaR с использованием сэмплов нормализующего потока.

    CVaR_α = E[X | X ≤ VaR_α]
    """
    # Генерируем сэмплы из изученного распределения
    z = torch.randn(n_samples, model.dim)
    samples = model.forward(z)

    # Находим порог VaR
    var = np.percentile(samples, alpha * 100)

    # Среднее сэмплов ниже VaR
    cvar = samples[samples <= var].mean()

    return var, cvar
```

### 4. Генерация синтетических данных

Генерация реалистичных сценариев доходности для:
- Стресс-тестирования
- Бэктестинга на большем объеме данных
- Обучения других моделей
- Симуляций Монте-Карло

```python
def generate_synthetic_returns(model, n_scenarios, conditioning=None):
    """
    Генерация синтетических сценариев доходности из изученного распределения.
    """
    # Сэмплируем из базового распределения
    z = torch.randn(n_scenarios, model.dim)

    # Преобразуем через поток
    if conditioning is not None:
        # Условная генерация (напр., режим высокой волатильности)
        synthetic_returns = model.forward(z, conditioning)
    else:
        synthetic_returns = model.forward(z)

    return synthetic_returns
```

### 5. Моделирование хвостового риска

```
┌─────────────────────────────────────────────────────────────┐
│              СРАВНЕНИЕ ХВОСТОВОГО РИСКА                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Вероятность дневной доходности -10%:                      │
│                                                             │
│   Гауссово (σ=2%):  P(X < -10%) = 3 × 10⁻⁷  (очень редко)  │
│   Исторические:     P(X < -10%) ≈ 0.1%      (бывает!)      │
│   Норм. поток:      P(X < -10%) ≈ 0.08%     (точно!)       │
│                                                             │
│   Поток изучает ИСТИННОЕ поведение хвостов!                 │
│                                                             │
│                    Гауссово                                  │
│          │    ***       vs       Изученный поток            │
│          │  *******                   ***                   │
│     P(x) │ *********                ******                  │
│          │***********              ********                 │
│          │           *            **********                │
│          └──────────────┘         ──────────┘              │
│           тонкие хвосты           тяжелые хвосты            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Дополнительные архитектуры потоков

### NICE (Non-linear Independent Components Estimation)

Простейшая архитектура потоков с аддитивной связью:

```python
# Аддитивный связывающий слой
def nice_forward(x, mask):
    x1, x2 = x * mask, x * (1 - mask)
    y1 = x1
    y2 = x2 + neural_net(x1)  # Аддитивное преобразование
    return y1 + y2

# Обратное преобразование тривиально!
def nice_inverse(y, mask):
    y1, y2 = y * mask, y * (1 - mask)
    x1 = y1
    x2 = y2 - neural_net(y1)  # Просто вычитаем
    return x1 + x2
```

### Glow (Generative Flow с обратимыми 1x1 свёртками)

Более выразительная архитектура, комбинирующая три компонента:

```
Блок Glow:
├── ActNorm: Обучаемая нормализация активаций
├── 1x1 свёртка: Обучаемая перестановка
└── Аффинная связь: Преобразование в стиле RealNVP

Многомасштабная архитектура:
Уровень 1: [Блок потока x K] → Разделение
Уровень 2: [Блок потока x K] → Разделение
Уровень L: [Блок потока x K] → Финальный z
```

### ActNorm (нормализация активаций)

Инициализация, зависящая от данных, стабилизирующая обучение:

```python
class ActNorm(nn.Module):
    """Нормализация активаций с data-dependent инициализацией"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.scale = nn.Parameter(torch.ones(1, dim))
        self.bias = nn.Parameter(torch.zeros(1, dim))
        self.initialized = False

    def initialize(self, x):
        """Инициализация на основе данных"""
        with torch.no_grad():
            mean = x.mean(dim=0, keepdim=True)
            std = x.std(dim=0, keepdim=True)
            self.bias.data = -mean
            self.scale.data = 1.0 / (std + 1e-6)
            self.initialized = True

    def forward(self, x):
        if not self.initialized:
            self.initialize(x)
        y = (x + self.bias) * self.scale
        log_det = torch.log(torch.abs(self.scale)).sum() * x.shape[0]
        return y, log_det

    def inverse(self, y):
        x = y / self.scale - self.bias
        return x
```

### Flow Matching (современный подход)

Более новая, упрощённая парадигма обучения для непрерывных нормализующих потоков:

```python
class FlowMatchingTrader:
    """Современный подход flow matching для торговых сигналов"""

    def __init__(self, vector_field_net):
        self.v_net = vector_field_net

    def flow_matching_loss(self, x0, x1):
        """
        Целевая функция flow matching
        x0: сэмплы шума (базовое распределение)
        x1: сэмплы данных (рыночные признаки)
        """
        t = torch.rand(x0.shape[0], 1)
        xt = (1 - t) * x0 + t * x1
        ut = x1 - x0
        vt = self.v_net(xt, t)
        loss = ((vt - ut) ** 2).mean()
        return loss

    def sample(self, num_samples, steps=100):
        """Генерация сэмплов через ODE интеграцию"""
        x = torch.randn(num_samples, self.dim)
        dt = 1.0 / steps
        for t in torch.linspace(0, 1, steps):
            v = self.v_net(x, t.expand(num_samples, 1))
            x = x + v * dt
        return x
```

---

## Торговые приложения: поток ордеров и микроструктура

### Предсказание потока ордеров

```python
class OrderFlowPredictor:
    """Предсказание потока ордеров с условной моделью потока"""

    def __init__(self, flow_model, context_encoder):
        self.flow = flow_model
        self.encoder = context_encoder

    def predict(self, market_context, num_samples=1000):
        context = self.encoder(market_context)
        z = torch.randn(num_samples, self.flow.latent_dim)
        predictions = self.flow.inverse(z, context)
        return {
            'expected_flow': predictions.mean(dim=0),
            'uncertainty': predictions.std(dim=0),
            'samples': predictions
        }
```

### Моделирование микроструктуры рынка

```python
class MicrostructureFlow:
    """Моделирование динамики стакана заявок нормализующими потоками"""

    def compute_likelihood(self, order_book_state):
        """Вычисление лог-правдоподобия конфигурации стакана"""
        z, log_det = self.flow.forward(order_book_state)
        log_pz = self.base_dist.log_prob(z).sum(dim=-1)
        return log_pz + log_det

    def detect_anomaly(self, order_book_state, threshold=-10.0):
        """Обнаружение необычных конфигураций стакана"""
        log_px = self.compute_likelihood(order_book_state)
        return log_px < threshold

    def simulate_book_evolution(self, initial_state, steps=100):
        """Симуляция эволюции стакана заявок"""
        states = [initial_state]
        for _ in range(steps):
            z, _ = self.flow.forward(states[-1])
            z_next = z + 0.01 * torch.randn_like(z)
            next_state = self.flow.inverse(z_next)
            states.append(next_state)
        return torch.stack(states)
```

### Обнаружение режимов в латентном пространстве

```python
class RegimeDetector:
    """Обнаружение рыночных режимов через латентное пространство потока"""

    def __init__(self, flow_model, n_regimes=4):
        self.flow = flow_model
        self.n_regimes = n_regimes
        self.clusterer = GaussianMixture(n_components=n_regimes)

    def fit_regimes(self, historical_data):
        """Обучение кластеров режимов на латентных представлениях"""
        z_latent, _ = self.flow.forward(historical_data)
        self.clusterer.fit(z_latent.detach().numpy())
        self.regime_labels = self._analyze_regimes(historical_data, z_latent)

    def detect_current_regime(self, current_data):
        """Определение текущего рыночного режима"""
        z, _ = self.flow.forward(current_data)
        regime = self.clusterer.predict(z.detach().numpy())
        probs = self.clusterer.predict_proba(z.detach().numpy())
        return {
            'regime': regime[0],
            'label': self.regime_labels[regime[0]],
            'confidence': probs.max(),
            'regime_probs': dict(zip(self.regime_labels, probs[0]))
        }
```

### Стресс-тестирование с потоками

```python
class FlowStressTester:
    """Генерация стресс-сценариев из областей низкого правдоподобия"""

    def __init__(self, flow_model):
        self.flow = flow_model

    def stress_test(self, portfolio, scenario_likelihood_threshold=-20.0):
        z_extreme = torch.randn(1000, self.flow.latent_dim) * 3
        extreme_scenarios = self.flow.inverse(z_extreme)
        log_probs = self.flow.log_prob(extreme_scenarios)

        mask = log_probs > scenario_likelihood_threshold
        stress_scenarios = extreme_scenarios[mask]

        impacts = [(scenario * portfolio.weights).sum().item()
                   for scenario in stress_scenarios]

        return {
            'scenarios': stress_scenarios,
            'impacts': impacts,
            'worst_case': min(impacts),
            'expected_shortfall': np.mean(sorted(impacts)[:int(len(impacts)*0.05)])
        }
```

---

## Требования к данным микроструктуры

Для высокочастотных торговых приложений потоковые модели выигрывают от богатых микроструктурных признаков:

```
Рыночные данные для потоковых моделей:
├── Высокочастотные данные (предпочтительно тиковые)
│   └── Поток ордеров, сделки, котировки
├── Снимки стакана заявок
│   └── Многоуровневые bid/ask с объёмами
├── Данные объёма
│   └── Разложение на покупку/продажу
└── Производные признаки
    ├── Дисбаланс потока ордеров (OFI)
    ├── Отклонение цены, взвешенное объёмом
    ├── Динамика спреда
    ├── Дисбаланс глубины
    ├── VPIN (Volume-synchronized PIN)
    └── Оценки лямбда Кайла
```

---

## Детали реализации

### Архитектура сети для Scale/Translation

```python
class CouplingNetwork(nn.Module):
    """
    Нейросеть для вычисления масштаба и сдвига в связывающих слоях.
    """
    def __init__(self, input_dim, hidden_dim=256, n_layers=3):
        super().__init__()

        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])

        # Выход: масштаб и сдвиг
        self.net = nn.Sequential(*layers)
        self.scale_net = nn.Linear(hidden_dim, input_dim)
        self.translation_net = nn.Linear(hidden_dim, input_dim)

        # Инициализация к тождественному преобразованию
        nn.init.zeros_(self.scale_net.weight)
        nn.init.zeros_(self.scale_net.bias)
        nn.init.zeros_(self.translation_net.weight)
        nn.init.zeros_(self.translation_net.bias)

    def forward(self, x):
        h = self.net(x)
        s = self.scale_net(h)
        t = self.translation_net(h)
        return s, t
```

### Аффинный связывающий слой

```python
class AffineCouplingLayer(nn.Module):
    """
    Аффинный связывающий слой как в RealNVP.
    """
    def __init__(self, dim, mask):
        super().__init__()
        self.dim = dim
        self.register_buffer('mask', mask)
        self.coupling_net = CouplingNetwork(dim // 2, hidden_dim=256)

    def forward(self, x):
        """Прямой проход: пространство данных -> латентное"""
        x_masked = x * self.mask
        s, t = self.coupling_net(x_masked)

        # Применяем преобразование к незамаскированной части
        y = x_masked + (1 - self.mask) * (x * torch.exp(s) + t)

        # Логарифм определителя якобиана
        log_det = (s * (1 - self.mask)).sum(dim=-1)

        return y, log_det

    def inverse(self, y):
        """Обратный проход: латентное -> пространство данных"""
        y_masked = y * self.mask
        s, t = self.coupling_net(y_masked)

        # Обратное преобразование
        x = y_masked + (1 - self.mask) * (y - t) * torch.exp(-s)

        # Логарифм определителя (отрицательный для обратного)
        log_det = -(s * (1 - self.mask)).sum(dim=-1)

        return x, log_det
```

### Полная модель нормализующего потока

```python
class NormalizingFlow(nn.Module):
    """
    Полный нормализующий поток для оценки плотности.
    """
    def __init__(self, dim, n_layers=8, hidden_dim=256):
        super().__init__()
        self.dim = dim

        # Создаем чередующиеся маски
        masks = []
        for i in range(n_layers):
            mask = torch.zeros(dim)
            mask[:dim//2] = 1.0 if i % 2 == 0 else 0.0
            mask[dim//2:] = 0.0 if i % 2 == 0 else 1.0
            masks.append(mask)

        # Стекаем связывающие слои
        self.layers = nn.ModuleList([
            AffineCouplingLayer(dim, masks[i])
            for i in range(n_layers)
        ])

        # Базовое распределение
        self.register_buffer('base_mean', torch.zeros(dim))
        self.register_buffer('base_std', torch.ones(dim))

    def forward(self, z):
        """Преобразование из латентного пространства в данные"""
        x = z
        for layer in self.layers:
            x, _ = layer.inverse(x)
        return x

    def inverse(self, x):
        """Преобразование из данных в латентное пространство"""
        z = x
        total_log_det = 0
        for layer in reversed(self.layers):
            z, log_det = layer(z)
            total_log_det += log_det
        return z, total_log_det

    def log_prob(self, x):
        """Вычисление логарифма вероятности данных"""
        z, log_det = self.inverse(x)
        log_pz = -0.5 * (z**2 + np.log(2*np.pi)).sum(dim=-1)
        return log_pz + log_det

    def sample(self, n_samples):
        """Генерация сэмплов из изученного распределения"""
        z = torch.randn(n_samples, self.dim)
        return self.forward(z)
```

### Конфигурация обучения

```yaml
model:
  dim: 1  # Одномерные доходности (или размерность портфеля)
  n_layers: 8
  hidden_dim: 256
  activation: "relu"
  use_batch_norm: true

training:
  batch_size: 256
  learning_rate: 0.0001
  weight_decay: 0.0001
  max_epochs: 500
  early_stopping_patience: 20
  gradient_clip: 1.0

data:
  train_split: 0.7
  val_split: 0.15
  test_split: 0.15
  lookback_window: 252  # 1 год дневных данных
  returns_type: "log"  # логарифмические доходности
  standardize: true
```

## Метрики риска с нормализующими потоками

### Расчет VaR

```python
def compute_var_flow(model, alpha_levels=[0.01, 0.05, 0.10], n_samples=100000):
    """
    Вычисление Value at Risk на нескольких уровнях доверия.
    """
    # Генерируем сэмплы
    samples = model.sample(n_samples).detach().numpy().flatten()

    var_results = {}
    for alpha in alpha_levels:
        var = np.percentile(samples, alpha * 100)
        var_results[f'VaR_{int((1-alpha)*100)}'] = var

    return var_results
```

### CVaR/Expected Shortfall

```python
def compute_cvar_flow(model, alpha=0.05, n_samples=100000):
    """
    Вычисление Conditional VaR (Expected Shortfall).
    """
    samples = model.sample(n_samples).detach().numpy().flatten()
    var = np.percentile(samples, alpha * 100)
    cvar = samples[samples <= var].mean()
    return var, cvar
```

### Хвостовая вероятность

```python
def compute_tail_probability(model, threshold, n_samples=100000):
    """
    Вычисление вероятности доходности ниже порога.
    P(X < threshold)
    """
    samples = model.sample(n_samples).detach().numpy().flatten()
    tail_prob = (samples < threshold).mean()
    return tail_prob
```

## Интеграция с торговой стратегией

### Генерация сигналов на основе плотности

```python
def generate_density_signals(model, current_return, historical_returns):
    """
    Генерация торговых сигналов на основе позиции в плотности.

    Если текущая доходность в области низкой вероятности,
    ожидаем возврат к среднему.
    """
    # Вычисляем лог-вероятность текущей доходности
    log_prob = model.log_prob(torch.tensor([[current_return]])).item()

    # Вычисляем перцентиль текущей доходности
    samples = model.sample(100000).numpy().flatten()
    percentile = (samples < current_return).mean()

    # Логика сигналов
    if percentile < 0.05:  # Экстремально низкая доходность
        return Signal("LONG", confidence=1 - percentile,
                      reason="Экстремально негативная доходность, ожидаем отскок")
    elif percentile > 0.95:  # Экстремально высокая доходность
        return Signal("SHORT", confidence=percentile,
                      reason="Экстремально позитивная доходность, ожидаем откат")
    else:
        return Signal("NEUTRAL", confidence=0.5)
```

### Управление портфельным риском

```python
class FlowBasedRiskManager:
    """
    Риск-менеджер на основе нормализующего потока для размера позиции.
    """
    def __init__(self, flow_model, max_var_pct=0.02):
        self.model = flow_model
        self.max_var = max_var_pct

    def compute_position_size(self, capital, confidence=0.99):
        """
        Размер позиции так, чтобы 99% VaR не превышал max_var_pct.
        """
        # Получаем VaR из потока
        var_99, _ = compute_cvar_flow(self.model, alpha=1-confidence)

        # Размер позиции такой, что потеря при VaR = max_var_pct от капитала
        position_size = (self.max_var * capital) / abs(var_99)

        return position_size
```

## Ключевые метрики

### Производительность модели

- **Negative Log-Likelihood (NLL)**: Чем меньше, тем лучше (качество оценки плотности)
- **Bits per Dimension (BPD)**: NLL / (dim * log(2))
- **Тест Колмогорова-Смирнова**: Сравнение изученного и эмпирического распределения
- **QQ Plot**: Визуальная проверка соответствия распределений

### Точность метрик риска

- **Бэктестинг VaR**: Подсчет нарушений (должен соответствовать уровню доверия)
- **Точность CVaR**: Сравнение предсказанных и реализованных хвостовых потерь
- **Тест Купека**: Статистический тест точности VaR

### Торговая эффективность

- **Коэффициент Шарпа**: Доходность с поправкой на риск (цель > 1.5)
- **Коэффициент Сортино**: Доходность с поправкой на нисходящий риск
- **Максимальная просадка**: Наибольшее падение от пика до впадины
- **Коэффициент Калмара**: Доходность / Макс. просадка

## Сравнение с другими методами

| Аспект | Гауссово | GARCH | Историч. симуляция | Норм. поток |
|--------|----------|-------|-------------------|-------------|
| Тяжелые хвосты | Нет | Частично | Да | Да |
| Асимметрия | Нет | Нет | Да | Да |
| Мультимодальность | Нет | Нет | Ограничено | Да |
| Обобщение | Плохо | Средне | Плохо | Хорошо |
| Точная плотность | Да | Приближение | Нет | Да |
| Синт. данные | Легко | Средне | Ограничено | Легко |
| Вычисл. стоимость | Низкая | Низкая | Низкая | Средняя |

## Сравнение с другими генеративными моделями

### vs. VAE

- **VAE**: Приблизительный постериор, обучение через ELBO, потери реконструкции
- **Поток**: Точное правдоподобие, идеальная реконструкция, без отдельного энкодера

### vs. GAN

- **GAN**: Нет плотности, коллапс мод, состязательное обучение
- **Поток**: Точная плотность, стабильное обучение, без дискриминатора

### vs. Диффузионные модели

- **Диффузия**: Медленная генерация, нет точного правдоподобия, высокое качество генерации
- **Поток**: Быстрая генерация, точное правдоподобие, более простая архитектура

| Аспект | Традиционные модели | Потоковые модели |
|--------|-------------------|-----------------|
| Правдоподобие | Приблизительное (VAE) или нет (GAN) | Точное вычисление |
| Реконструкция | С потерями | Идеальная (обратимая) |
| Обнаружение аномалий | Пороги на признаках | Принципиальная оценка плотности |
| Неопределённость | Часто отсутствует | Естественная из плотности |
| Интерпретируемость | Чёрный ящик | Структура латентного пространства |
| Качество сэмплов | Коллапс мод (GAN) | Стабильное обучение |

---

## Продвинутые темы

### 1. Условные нормализующие потоки

Условие потока на внешних факторах (режим волатильности, рыночные условия):

```python
class ConditionalFlow(nn.Module):
    def __init__(self, dim, cond_dim, n_layers=8):
        # Сеть кондиционирования
        self.cond_net = nn.Sequential(
            nn.Linear(cond_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )

        # Связывающие сети принимают условие на вход
        self.layers = nn.ModuleList([
            ConditionalCouplingLayer(dim, cond_dim=64)
            for _ in range(n_layers)
        ])
```

### 2. Многомерные потоки для портфеля

Моделирование совместного распределения нескольких активов:

```python
# Вместо моделирования каждого актива отдельно
# Моделируем полную ковариационную структуру
flow = NormalizingFlow(dim=10)  # 10 активов

# Совместные сэмплы учитывают корреляции
joint_samples = flow.sample(1000)  # [1000, 10]

# VaR портфеля учитывает диверсификацию
portfolio_returns = joint_samples @ weights
portfolio_var = np.percentile(portfolio_returns, 5)
```

### 3. Динамические потоки

Обновление параметров потока при изменении рыночных условий:

```python
class AdaptiveFlow:
    def __init__(self, base_flow, adaptation_rate=0.01):
        self.flow = base_flow
        self.rate = adaptation_rate

    def update(self, new_data):
        """Онлайн-обновление с новыми наблюдениями"""
        loss = -self.flow.log_prob(new_data).mean()
        loss.backward()

        with torch.no_grad():
            for param in self.flow.parameters():
                param -= self.rate * param.grad
                param.grad.zero_()
```

## Продакшн-соображения

```
Конвейер инференса:
├── Сбор данных (Bybit через CCXT)
│   └── OHLCV данные в реальном времени
├── Расчет доходностей
│   └── Логарифмические доходности со скользящей статистикой
├── Инференс модели
│   └── Оценка плотности / генерация сэмплов
├── Расчет риска
│   └── VaR, CVaR, хвостовые вероятности
├── Генерация сигналов
│   └── На основе позиции в плотности
└── Исполнение
    └── Размер позиции из модели риска

Бюджет задержки:
├── Получение данных: ~50мс (REST API)
├── Предобработка: ~1мс
├── Инференс потока: ~5мс (GPU)
├── Расчет риска: ~10мс (MC сэмплирование)
├── Генерация сигналов: ~1мс
└── Всего: ~70мс
```

## Структура директории

```
332_normalizing_flows_finance/
├── README.md                    # Основной файл (английский)
├── README.ru.md                 # Этот файл (русский)
├── readme.simple.md             # Объяснение для начинающих (английский)
├── readme.simple.ru.md          # Объяснение для начинающих (русский)
├── python/                      # Python реализация
│   ├── __init__.py
│   ├── flows.py                 # Модели нормализующих потоков
│   ├── layers.py                # Связывающие слои
│   ├── risk_metrics.py          # Расчеты VaR, CVaR
│   ├── data_fetcher.py          # Данные Bybit через CCXT
│   ├── training.py              # Цикл обучения
│   └── examples/
│       ├── density_estimation.py
│       ├── var_calculation.py
│       └── synthetic_generation.py
└── rust_normalizing_flows/      # Rust реализация
    ├── Cargo.toml
    ├── src/
    │   ├── lib.rs
    │   ├── api/                 # API клиент Bybit
    │   ├── flows/               # Реализации потоков
    │   ├── risk/                # Метрики риска
    │   └── utils/               # Утилиты
    └── examples/
        ├── fetch_data.rs
        ├── train_flow.rs
        └── compute_var.rs
```

## Ссылки

1. **NICE: Non-linear Independent Components Estimation** (Dinh et al., 2014)
   - https://arxiv.org/abs/1410.8516

2. **Variational Inference with Normalizing Flows** (Rezende & Mohamed, 2015)
   - https://arxiv.org/abs/1505.05770

3. **Density Estimation using Real-NVP** (Dinh et al., 2016)
   - https://arxiv.org/abs/1605.08803

4. **Masked Autoregressive Flow for Density Estimation** (Papamakarios et al., 2017)
   - https://arxiv.org/abs/1705.07057

5. **Glow: Generative Flow with Invertible 1x1 Convolutions** (Kingma & Dhariwal, 2018)
   - https://arxiv.org/abs/1807.03039

6. **Neural Ordinary Differential Equations** (Chen et al., 2018)
   - https://arxiv.org/abs/1806.07366

7. **Neural Spline Flows** (Durkan et al., 2019)
   - https://arxiv.org/abs/1906.04032

8. **Normalizing Flows for Probabilistic Modeling and Inference** (Papamakarios et al., 2019)
   - https://arxiv.org/abs/1912.02762

9. **Flow Matching for Generative Modeling** (Lipman et al., 2022)
   - https://arxiv.org/abs/2210.02747

## Уровень сложности

**Продвинутый** - Требуется понимание:
- Теории вероятностей и оценки плотности
- Формулы замены переменных
- Определителей якобианов
- Основ глубокого обучения
- Финансовых метрик риска (VaR, CVaR)

## Отказ от ответственности

Эта глава предназначена **только для образовательных целей**. Торговля криптовалютами связана с существенным риском. Описанные здесь стратегии и модели риска должны быть тщательно проверены перед любым практическим применением. Прошлые результаты не гарантируют будущих. Всегда консультируйтесь с финансовыми специалистами перед принятием инвестиционных решений.
