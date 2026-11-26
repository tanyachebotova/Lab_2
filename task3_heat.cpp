#include <algorithm>
#include <cerrno>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <omp.h>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
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

// Строкам сетки соответствует y, столбцам соответствует x
struct Plate {
    int m, stride;
    double c, amplitude;
    bool analytic;
    std::vector<double> u, h2f, sine;
};

// Задание температуры на границе и выбор источника тепла
Plate make_plate(int m, double c, const char* source) {
    const int stride = m + 1;
    std::size_t cells = static_cast<std::size_t>(stride) * stride;
    bool analytic = source[0] != '@';
    double amplitude = analytic ? real(source) : 0;
    // Начальная температура во всех узлах равна c
    Plate plate{m, stride, c, amplitude, analytic, std::vector<double>(cells, c),
                std::vector<double>(cells), std::vector<double>(stride)};
    double pi = std::acos(-1.0), h2 = 1.0 / (static_cast<double>(m) * m);
    std::ifstream file;
    if (analytic) {
        // Для проверки используется синусоидальное поле с известным точным решением
        for (int i = 1; i < m; ++i) plate.sine[i] = std::sin(pi * i / m);
    } else {
        // Значения источника читаются по строкам внутренних узлов
        file.open(source + 1);
        if (!file) throw std::runtime_error("Не удалось открыть файл источника.");
    }
    for (int y = 1; y < m; ++y) {
        for (int x = 1; x < m; ++x) {
            double f;
            if (analytic) f = -2*pi*pi*amplitude * plate.sine[x] * plate.sine[y];
            else if (!(file >> f) || !std::isfinite(f))
                throw std::runtime_error("В файле источника нужны конечные числа для всех узлов.");
            // Произведение h² и f вычисляется заранее для каждого узла
            if (!std::isfinite(h2*f)) throw std::runtime_error("Значение источника слишком велико.");
            plate.h2f[static_cast<std::size_t>(y)*stride+x] = h2*f;
        }
    }
    if (!analytic) {
        std::string extra;
        if (file >> extra) throw std::runtime_error("После таблицы источника остались лишние значения.");
    }
    return plate;
}

// Построчный обход блока с поиском максимального изменения температуры
double update_block(Plate& p, int row, int col, int block) {
    const int y0 = 1 + row*block, x0 = 1 + col*block;
    // Крайний блок ограничен внутренними узлами сетки
    const int y1 = std::min(y0+block, p.m), x1 = std::min(x0+block, p.m);
    double change = 0;
    for (int y = y0; y < y1; ++y) {
        std::size_t k = static_cast<std::size_t>(y)*p.stride + x0;
        for (int x = x0; x < x1; ++x, ++k) {
            // Сверху и слева берутся новые значения, снизу и справа остаются значения прошлого прохода
            double next = 0.25 * (p.u[k-p.stride]+p.u[k+p.stride]+p.u[k-1]+p.u[k+1]-p.h2f[k]);
            double step = std::abs(next - p.u[k]);
            change = std::isfinite(step) ? std::max(change, step) : INFINITY;
            p.u[k] = next;
        }
    }
    return change;
}

// Проходы по диагоналям повторяются до заданной точности
int solve(Plate& p, int block, double eps, int threads, double& delta) {
    int blocks = (p.m - 2) / block + 1;
    constexpr int limit = 2000000;
    std::vector<double> changes(threads);
    int sweeps = 0;
    bool done = false;
    // Один набор потоков используется на всех итерациях
    #pragma omp parallel
    {
        do {
            double local_change = 0;
            for (int wave = 0; wave < 2*blocks-1; ++wave) {
                int first = std::max(0, wave-blocks+1), last = std::min(wave, blocks-1);
                // Следующая диагональ доступна после завершения всех блоков текущей
                #pragma omp for schedule(static)
                for (int row = first; row <= last; ++row)
                    local_change = std::max(local_change, update_block(p, row, wave-row, block));
            }
            // Каждый поток сохраняет свой максимум за итерацию
            changes[omp_get_thread_num()] = local_change;
            #pragma omp barrier
            #pragma omp single
            {
                // Общий максимум определяет момент остановки
                delta = *std::max_element(changes.begin(), changes.end());
                ++sweeps;
                done = delta <= eps || !std::isfinite(delta) || sweeps == limit;
            }
        } while (!done);
    }
    return sweeps;
}

// Проверка невязки уравнения и ошибки относительно точного решения
std::pair<double,double> errors(const Plate& p) {
    double residual = 0, exact_error = 0;
    double inv_h2 = static_cast<double>(p.m)*p.m;
    for (int y = 1; y < p.m; ++y) {
        for (int x = 1; x < p.m; ++x) {
            auto k = static_cast<std::size_t>(y)*p.stride+x;
            double diff = p.u[k-p.stride]+p.u[k+p.stride]+p.u[k-1]+p.u[k+1]-4*p.u[k]-p.h2f[k];
            residual = std::max(residual, std::abs(diff)*inv_h2);
            if (p.analytic) {
                double exact = p.c + p.amplitude*p.sine[x]*p.sine[y];
                exact_error = std::max(exact_error, std::abs(p.u[k]-exact));
            }
        }
    }
    return {residual, exact_error};
}

// Запись координат и температур всех узлов вместе с границей
void save_grid(const Plate& p, const char* path) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Не удалось создать CSV с температурой.");
    out << "x,y,u\n" << std::setprecision(17);
    for (int y = 0; y <= p.m; ++y)
        for (int x = 0; x <= p.m; ++x)
            out << static_cast<double>(x)/p.m << ',' << static_cast<double>(y)/p.m << ','
                << p.u[static_cast<std::size_t>(y)*p.stride + x] << '\n';
    out.close();
    if (!out) throw std::runtime_error("Не удалось записать сетку полностью.");
}

// Чтение параметров пластины и расчёт температуры
int main(int argc, char** argv) try {
    if (argc < 5 || argc > 8)
        throw std::invalid_argument("Запуск: heat ПОТОКИ M БЛОК EPS [c [A|@файл_f [поле.csv]]]");
    int threads = configure_threads(argv[1]);
    int m = integer(argv[2], 2, std::numeric_limits<int>::max()/4);
    int block = std::min(integer(argv[3], 1, std::numeric_limits<int>::max()/2), m-1);
    double eps = real(argv[4]);
    if (eps <= 0) throw std::invalid_argument("EPS должна быть положительной.");
    double c = argc > 5 ? real(argv[5]) : 20.0;
    Plate plate = make_plate(m, c, argc > 6 ? argv[6] : "10");
    // Замер времени от первого прохода до остановки
    double start = omp_get_wtime(), delta = 0;
    int sweeps = solve(plate, block, eps, threads, delta);
    double seconds = omp_get_wtime() - start;
    if (!std::isfinite(delta) || delta > eps)
        throw std::runtime_error("Точность не достигнута за отведённое число итераций либо произошло переполнение.");
    // Вывод результатов и запись сетки при указанном имени файла
    auto [residual, exact_error] = errors(plate);
    std::size_t center = static_cast<std::size_t>(m/2)*plate.stride + m/2;
    std::printf("threads=%d M=%d block=%d sweeps=%d delta=%.12g residual=%.12g center=%.12g ",
                threads, m, block, sweeps, delta, residual, plate.u[center]);
    if (plate.analytic) std::printf("max_error=%.12g ", exact_error);
    std::printf("seconds=%.9f\n", seconds);
    if (argc == 8) save_grid(plate, argv[7]);
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
}
