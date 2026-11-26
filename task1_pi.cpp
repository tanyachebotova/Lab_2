#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <omp.h>
#include <stdexcept>
#include <string>
#include <system_error>

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

// Проверка начального числа на нечётность
inline std::uint64_t read_seed(const char* text) {
    auto seed = integer<std::uint64_t>(text, 1, mcg_mask);
    if ((seed & 1) == 0) throw std::invalid_argument("Для MCG59 требуется нечётный seed.");
    return seed;
}

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

// Оценка числа π по доле точек внутри круга
int main(int argc, char** argv) try {
    // Чтение числа потоков, числа точек и начального состояния генератора
    if (argc < 3 || argc > 4)
        throw std::invalid_argument("Запуск: pi ПОТОКИ N [НЕЧЁТНЫЙ_SEED]");
    int threads = configure_threads(argv[1]);
    auto n = integer<std::int64_t>(argv[2], 1, INT64_C(1) << 56);
    auto seed = argc == 4 ? read_seed(argv[3]) : default_seed;

    // Последний блок может содержать меньше точек
    std::int64_t blocks = (n + sample_block - 1) / sample_block;
    std::int64_t hits = 0;
    double start = omp_get_wtime();
    #pragma omp parallel for schedule(static) reduction(+:hits)
    for (std::int64_t block = 0; block < blocks; ++block) {
        // На каждую точку предыдущих блоков приходится два шага генератора
        std::int64_t first = block * sample_block;
        std::int64_t end = std::min(n, first + sample_block);
        std::uint64_t state = mcg_skip(seed, 2 * first);
        for (auto i = first; i < end; ++i) {
            // Координаты точки в квадрате со стороной 2
            double x = 2.0 * mcg_uniform(state) - 1.0;
            double y = 2.0 * mcg_uniform(state) - 1.0;
            // Точка внутри круга или на границе, если сумма квадратов координат не превышает единицу
            hits += (x*x + y*y <= 1.0);
        }
    }
    // Умножение доли попаданий на четыре и вычисление погрешности
    double estimate = 4.0 * static_cast<double>(hits) / n;
    double seconds = omp_get_wtime() - start;
    std::printf("threads=%d N=%lld seed=%llu hits=%lld pi=%.12f abs_error=%.12g seconds=%.9f\n",
                threads, (long long)n, (unsigned long long)seed, (long long)hits,
                estimate, std::abs(estimate - std::acos(-1.0)), seconds);
    return 0;
} catch (const std::exception& e) {
    // Сообщение об ошибке ввода выводится в stderr
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
}
