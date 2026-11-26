#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>
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

// Координаты точки хранятся в типе double
struct Point { double x, y; };

// При сложении учитывается перенос в следующий разряд
using Words = std::array<std::uint64_t, 68>;
inline void add_word(Words& sum, int pos, std::uint64_t word) {
    while (word != 0) {
        sum[pos] += word;
        word = (sum[pos] < word);
        ++pos;
    }
}

// Знак произведения определяет выбор части суммы
inline void exact_product(Words& positive, Words& negative, double a, double b, bool subtract) {
    if (a == 0 || b == 0) return;
    // Каждая координата представляется через мантиссу и степень двойки
    int ea, eb;
    auto ma = static_cast<std::uint64_t>(std::ldexp(std::frexp(std::abs(a), &ea), 53));
    auto mb = static_cast<std::uint64_t>(std::ldexp(std::frexp(std::abs(b), &eb), 53));
    __uint128_t product = static_cast<__uint128_t>(ma) * mb;
    // Сдвиг на 2252 учитывает даже самые маленькие произведения
    int shift = ea + eb - 106 + 2252;
    Words& sum = ((std::signbit(a) != std::signbit(b)) != subtract) ? negative : positive;
    for (int half = 0; half < 2; ++half) {
        // Обе половины произведения записываются с нужным сдвигом
        std::uint64_t part = static_cast<std::uint64_t>(product);
        int pos = shift / 64 + half;
        int bits = shift % 64;
        add_word(sum, pos, part << bits);
        if (bits != 0) add_word(sum, pos + 1, part >> (64 - bits));
        product >>= 64;
    }
}

// Знак определителя задаёт направление поворота по трём точкам
inline int turn(Point a, Point b, Point c) {
    // Проверка знака с учётом погрешности округления
    double left = (a.x - c.x) * (b.y - c.y);
    double right = (a.y - c.y) * (b.x - c.x);
    double det = left - right;
    double scale = std::abs(left) + std::abs(right);
    constexpr double eps = std::numeric_limits<double>::epsilon();
    if (std::isfinite(scale) && scale > std::numeric_limits<double>::min() / eps &&
        std::abs(det) > 8 * eps * scale)
        return det > 0 ? 1 : -1;
    // Точки с одинаковым x или одинаковым y лежат на одной прямой
    if ((a.x == b.x && b.x == c.x) || (a.y == b.y && b.y == c.y)) return 0;
    // При сомнительном знаке шесть произведений вычисляются точно
    Words pos{}, neg{};
    exact_product(pos, neg, a.x, b.y, false);
    exact_product(pos, neg, b.x, c.y, false);
    exact_product(pos, neg, c.x, a.y, false);
    exact_product(pos, neg, a.y, b.x, true);
    exact_product(pos, neg, b.y, c.x, true);
    exact_product(pos, neg, c.y, a.x, true);
    // Сравнение начинается со старших разрядов
    for (int i = static_cast<int>(pos.size()) - 1; i >= 0; --i) {
        if (pos[i] != neg[i]) return pos[i] > neg[i] ? 1 : -1;
    }
    return 0;
}

// Обе цепи хранят индексы точек в порядке слева направо
using Index = std::uint32_t;
using Chain = std::vector<Index>;
struct Hull { Chain lower, upper; };
constexpr Index leaf_size = 64;
constexpr Index parallel_cutoff = 16384;

// Сравнение по x, при равенстве по y
bool before(Point a, Point b) {
    return a.x < b.x || (a.x == b.x && a.y < b.y);
}

// Проверка совпадения обеих координат
bool same(Point a, Point b) {
    return a.x == b.x && a.y == b.y;
}

// Удаление лишних вершин при добавлении новой точки
void append_vertex(const std::vector<Point>& p, Chain& chain, Index next, int direction) {
    while (chain.size() >= 2 &&
           direction * turn(p[chain[chain.size()-2]], p[chain.back()], p[next]) <= 0)
        chain.pop_back();
    chain.push_back(next);
}

// Соединение двух цепей по общей касательной
Chain join(const std::vector<Point>& p, Chain left, const Chain& right, int direction) {
    // Поиск начинается с ближайших концов без повтора граничной точки
    std::size_t i = left.size() - 1;
    std::size_t j = same(p[left[i]], p[right.front()]) ? 1 : 0;
    if (j == right.size()) return left;
    // Конец касательной сдвигается, пока соседняя точка выступает за прямую
    for (;;) {
        while (i > 0 && direction * turn(p[left[i-1]], p[left[i]], p[right[j]]) <= 0) --i;
        if (j + 1 == right.size() ||
            direction * turn(p[left[i]], p[right[j]], p[right[j+1]]) > 0) break;
        ++j;
    }
    // Начало левой цепи соединяется с концом правой
    left.resize(i + 1);
    left.insert(left.end(), right.begin() + j, right.end());
    return left;
}

// Построение оболочки от first до last без включения last
Hull make_hull(std::vector<Point>& p, Index first, Index last) {
    if (last - first <= leaf_size) {
        // Для небольшого набора достаточно сортировки и прохода по точкам
        std::sort(p.begin()+first, p.begin()+last, before);
        Hull result;
        for (Index i = first; i < last; ++i) {
            if (i != first && same(p[i], p[i-1])) continue;
            append_vertex(p, result.lower, i, +1);
            append_vertex(p, result.upper, i, -1);
        }
        return result;
    }
    // Медиана делит точки пополам по x с учётом y при равенстве x
    Index middle = first + (last - first) / 2;
    std::nth_element(p.begin()+first, p.begin()+middle, p.begin()+last, before);
    Hull left, right;
    if (last - first >= parallel_cutoff && omp_get_num_threads() > 1) {
        // Левая половина передаётся отдельной задаче, правая обрабатывается в текущей
        #pragma omp task shared(p,left) firstprivate(first,middle)
        left = make_hull(p, first, middle);
        right = make_hull(p, middle, last);
        // Перед объединением должны быть готовы обе половины
        #pragma omp taskwait
    } else {
        // Для малых наборов отдельная задача не создаётся
        left = make_hull(p, first, middle);
        right = make_hull(p, middle, last);
    }
    // У верхней и нижней цепей разные допустимые направления поворота
    return {join(p, std::move(left.lower), right.lower, +1),
            join(p, std::move(left.upper), right.upper, -1)};
}

// Чтение точек и вывод вершин против часовой стрелки
int main(int argc, char** argv) try {
    if (argc != 2) throw std::invalid_argument("Запуск: hull ПОТОКИ < points.txt");
    int threads = configure_threads(argv[1]);
    // Чтение координат через scanf сохраняет очень малые значения
    long long count;
    if (std::scanf("%lld", &count) != 1 || count < 1 || count > 100000000)
        throw std::invalid_argument("Требуется 1 ≤ N ≤ 100000000.");
    std::vector<Point> points(static_cast<std::size_t>(count));
    for (auto& point : points) {
        if (std::scanf("%lf%lf", &point.x, &point.y) != 2 ||
            !std::isfinite(point.x) || !std::isfinite(point.y))
            throw std::invalid_argument("Ожидались N пар конечных координат.");
    }
    // Деление начинается с полного набора точек
    Hull hull;
    double start = omp_get_wtime();
    #pragma omp parallel
    {
        #pragma omp single
        hull = make_hull(points, 0, static_cast<Index>(count));
    }
    // Нижняя цепь выводится слева направо, верхняя в обратном порядке без общих концов
    std::size_t inner_upper = hull.upper.size() > 2 ? hull.upper.size() - 2 : 0;
    double seconds = omp_get_wtime() - start;
    std::cout << hull.lower.size() + inner_upper << '\n' << std::setprecision(17);
    for (auto i : hull.lower) std::cout << points[i].x << ' ' << points[i].y << '\n';
    for (auto j = inner_upper; j > 0; --j) {
        auto point = points[hull.upper[j]];
        std::cout << point.x << ' ' << point.y << '\n';
    }
    // Время выводится в stderr отдельно от координат вершин
    std::cerr << "threads=" << threads << " N=" << count << " seconds="
              << std::fixed << std::setprecision(9) << seconds << '\n';
    return 0;
} catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
}
