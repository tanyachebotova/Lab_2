#include <algorithm>
#include <array>
#include <cerrno>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <omp.h>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#ifndef _OPENMP
#error "Требуется компилятор с включённым OpenMP"
#endif

// Проверка целого числа и допустимых границ
template<class T>
T integer(const char* text, T minimum, T maximum) {
    T value{};
    const char* end = text + std::strlen(text);
    auto result = std::from_chars(text, end, value);
    if (result.ec != std::errc{} || result.ptr != end || value < minimum || value > maximum)
        throw std::invalid_argument(std::string("Недопустимое целое число: ") + text);
    return value;
}

// Проверка записи вещественного числа
inline double real(const char* text) {
    char* end = nullptr;
    errno = 0;
    double value = std::strtod(text, &end);
    if (end == text || *end != '\0' || errno == ERANGE || !std::isfinite(value))
        throw std::invalid_argument(std::string("Недопустимое число: ") + text);
    return value;
}

// Проверка числа доступных потоков перед замером времени
inline int configure_threads(const char* text) {
    int requested = integer(text, 1, 4096);
    omp_set_dynamic(0);
    omp_set_num_threads(requested);
    int actual = 0;
    #pragma omp parallel
    {
        #pragma omp single
        actual = omp_get_num_threads();
    }
    if (actual != requested)
        throw std::runtime_error("OpenMP выделил другое число потоков. Проверьте OMP_THREAD_LIMIT.");
    return actual;
}

// Для MCG59 нужны множитель 13^13 и нечётное начальное число
constexpr std::uint64_t mcg_multiplier = UINT64_C(302875106592253);
constexpr std::uint64_t mcg_mask = (UINT64_C(1) << 59) - 1;
constexpr std::uint64_t default_seed = UINT64_C(1234567);
constexpr std::int64_t sample_block = 2048;

// Переход к началу блока быстрым возведением в степень
inline std::uint64_t mcg_skip(std::uint64_t state, std::uint64_t steps) {
    std::uint64_t factor = mcg_multiplier;
    while (steps > 0) {
        // Младшие 59 бит дают остаток по модулю 2^59
        if (steps & 1) state = (state * factor) & mcg_mask;
        factor = (factor * factor) & mcg_mask;
        steps >>= 1;
    }
    return state;
}

// Следующее число MCG59 переводится в интервал между нулём и единицей
inline double mcg_uniform(std::uint64_t& state) {
    state = (state * mcg_multiplier) & mcg_mask;
    // Середина ячейки лежит строго внутри интервала
    return (static_cast<double>(state >> 7) + 0.5) * 0x1p-52;
}

constexpr double inv_sqrt2 = 0.7071067811865475244;
constexpr double inv_sqrt2pi = 0.39894228040143267794;

// Вероятность того, что стандартная нормальная величина не превысит z
double normal_cdf(double z) { return 0.5 * std::erfc(-z * inv_sqrt2); }

// Начальная оценка квантиля уточняется двумя шагами Ньютона
double normal_quantile(double u) {
    // Меньшая из двух вероятностей позволяет сохранить точность в хвосте распределения
    double tail = std::min(u, 1.0-u);
    double t = std::sqrt(-2.0*std::log(tail));
    double z = t - ((0.010328*t + 0.802853)*t + 2.515517) /
                    (((0.001308*t + 0.189269)*t + 1.432788)*t + 1.0);
    for (int step = 0; step < 2; ++step) {
        // Плотность в текущей точке нужна для поправки Ньютона
        double density = inv_sqrt2pi * std::exp(-0.5*z*z);
        z += (0.5*std::erfc(z*inv_sqrt2) - tail) / density;
    }
    return u < 0.5 ? -z : z;
}

// Параметры акции и коэффициенты для повторных расчётов
struct Asset { double spot, sigma, drift, diffusion; };
struct Market { Asset first, second; double strike, rate, maturity, discount; int dims; };

// Расчёт коэффициентов по цене и волатильности акции без дивидендов
Asset asset(double spot, double sigma, double rate, double time) {
    if (spot <= 0 || sigma < 0) throw std::invalid_argument("Нужны S>0 и σ≥0.");
    return {spot, sigma, (rate-0.5*sigma*sigma)*time, sigma*std::sqrt(time)};
}

// Цена акции к концу срока для выбранного сценария
double terminal(const Asset& a, double z) { return a.spot*std::exp(a.drift + a.diffusion*z); }

// Выплата по одной акции или по более дорогой из двух
double payoff(const Market& m, double z1, double z2) {
    double last = terminal(m.first, z1);
    if (m.dims == 2) last = std::max(last, terminal(m.second, z2));
    return std::max(last-m.strike, 0.0);
}

// Контрольная цена по формуле Блэка и Шоулса
double black_scholes(const Market& m) {
    const auto& a = m.first;
    // При нулевом сроке или нулевой волатильности выплата известна заранее
    if (a.diffusion == 0) return std::max(a.spot-m.strike*m.discount, 0.0);
    double d1 = (std::log(a.spot/m.strike) + (m.rate+0.5*a.sigma*a.sigma)*m.maturity) / a.diffusion;
    double d2 = d1-a.diffusion;
    return a.spot*normal_cdf(d1) - m.strike*m.discount*normal_cdf(d2);
}

// Направления последовательности Соболя для двух координат
using Pair = std::array<std::uint64_t,2>;
using Directions = std::array<Pair,64>;
Directions sobol_directions() {
    Directions v{};
    // Для первой координаты используются степени одной второй, для второй полином x+1
    v[0] = {UINT64_C(1)<<63, UINT64_C(1)<<63};
    for (int bit = 1; bit < 64; ++bit)
        v[bit] = {v[bit-1][0] >> 1, v[bit-1][1] ^ (v[bit-1][1] >> 1)};
    return v;
}

// Поиск точки Соболя по номеру для перехода к началу блока
Pair sobol_at(std::uint64_t index, const Directions& v) {
    Pair point{};
    // Единичные биты кода Грея задают направления для операции XOR
    for (auto gray = index ^ (index >> 1); gray != 0; gray &= gray-1) {
        int bit = __builtin_ctzll(gray);
        point[0] ^= v[bit][0];
        point[1] ^= v[bit][1];
    }
    return point;
}

// Перевод координаты в открытый интервал между нулём и единицей
double sobol_uniform(std::uint64_t word) {
    return (static_cast<double>(word >> 12) + 0.5) * 0x1p-52;
}

// Расчёт средней выплаты по сценариям с учётом дисконтирования
double simulate(const Market& m, std::int64_t n, bool sobol) {
    std::int64_t blocks = (n+sample_block-1) / sample_block;
    std::vector<double> totals(static_cast<std::size_t>(blocks));
    const Directions v = sobol_directions();
    #pragma omp parallel for schedule(static)
    for (std::int64_t b = 0; b < blocks; ++b) {
        // Номер блока определяет начальную позицию в последовательности
        auto first = b*sample_block, end = std::min(n, first+sample_block);
        std::uint64_t state = 0;
        Pair point{};
        if (sobol) point = sobol_at(first,v);
        else state = mcg_skip(default_seed, m.dims*first);
        double local = 0;
        for (auto i = first; i < end; ++i) {
            double u1, u2 = 0.5;
            if (sobol) {
                // После чтения текущей точки Соболя вычисляется следующая
                u1 = sobol_uniform(point[0]);
                if (m.dims == 2) u2 = sobol_uniform(point[1]);
                int bit = __builtin_ctzll(static_cast<unsigned long long>(i+1));
                point[0] ^= v[bit][0];
                if (m.dims == 2) point[1] ^= v[bit][1];
            } else {
                // Для двух акций используются два последовательных числа генератора
                u1 = mcg_uniform(state);
                if (m.dims == 2) u2 = mcg_uniform(state);
            }
            // Равномерные числа преобразуются в нормальные отклонения для расчёта выплаты
            double z2 = m.dims == 2 ? normal_quantile(u2) : 0.0;
            local += payoff(m, normal_quantile(u1), z2);
        }
        totals[b] = local;
    }
    // Частичные суммы складываются в порядке блоков для воспроизводимости результата
    return m.discount * (std::accumulate(totals.begin(), totals.end(), 0.0) / n);
}

// Вычисление интеграла по значениям в серединах ячеек
double midpoint(const Market& m, std::int64_t n) {
    auto side = m.dims == 2 ? static_cast<std::int64_t>(std::llround(std::sqrt(n))) : n;
    if (m.dims == 2 && side*side != n)
        throw std::invalid_argument("Для двумерной квадратуры N должно быть полным квадратом.");
    // Границы интегрирования учитывают возможный разброс цены акции
    double bound = 9 + std::max(m.first.diffusion, m.second.diffusion);
    double h = 2*bound / side, sum = 0;
    #pragma omp parallel for schedule(static) reduction(+:sum)
    for (std::int64_t i = 0; i < side; ++i) {
        double z1 = -bound+(i+0.5)*h;
        double weight1 = h*inv_sqrt2pi*std::exp(-0.5*z1*z1);
        if (m.dims == 1) {
            sum += weight1*payoff(m, z1, 0);
        } else {
            // Для независимых акций веса двух координат перемножаются
            double row = 0;
            for (std::int64_t j = 0; j < side; ++j) {
                double z2 = -bound+(j+0.5)*h;
                double weight2 = h*inv_sqrt2pi*std::exp(-0.5*z2*z2);
                row += weight2*payoff(m, z1, z2);
            }
            sum += weight1*row;
        }
    }
    return m.discount*sum;
}

// Чтение параметров опциона и запуск выбранного метода
int main(int argc, char** argv) try {
    if (argc != 9 && argc != 11)
        throw std::invalid_argument("Запуск: option ПОТОКИ N mcg59|sobol|quad S0 K r sigma T [S2 sigma2]");
    int threads = configure_threads(argv[1]);
    auto n = integer<std::int64_t>(argv[2], 1, INT64_C(1)<<56);
    std::string method = argv[3];
    if (method != "mcg59" && method != "sobol" && method != "quad")
        throw std::invalid_argument("Метод должен быть mcg59, sobol или quad.");
    double k = real(argv[5]), r = real(argv[6]), t = real(argv[8]);
    if (k <= 0 || t < 0) throw std::invalid_argument("Требуется K>0 и T≥0.");
    // Вторая акция учитывается при заданных цене и волатильности
    Asset one = asset(real(argv[4]),real(argv[7]),r,t);
    Asset two = argc == 11 ? asset(real(argv[9]),real(argv[10]),r,t) : one;
    Market m{one,two,k,r,t,std::exp(-r*t),argc == 11 ? 2 : 1};
    // Замер времени включает вычисление цены и сложение частичных сумм
    double start = omp_get_wtime();
    double price = method == "quad" ? midpoint(m,n) : simulate(m,n,method == "sobol");
    double seconds = omp_get_wtime()-start;
    if (!std::isfinite(price)) throw std::runtime_error("Параметры модели привели к переполнению.");
    std::printf("threads=%d N=%lld dims=%d method=%s price=%.12f ",
                threads,(long long)n,m.dims,method.c_str(),price);
    if (m.dims == 1) {
        // Для одной акции оценка сравнивается с аналитической ценой
        double exact = black_scholes(m);
        std::printf("black_scholes=%.12f abs_error=%.12g ",exact,std::abs(price-exact));
    }
    std::printf("seconds=%.9f\n",seconds);
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
}
