import math
import random
import matplotlib.pyplot as plt


class GammaInteraction:
    """
    Класс для моделирования взаимодействия гамма-излучения с детектором NaI(Tl)
    методом Монте-Карло
    """

    def __init__(self):
        """Инициализация параметров детектора, физических констант и ввод данных"""

        self.NA = 6.022e23
        self.mc2 = 0.511
        self.Z_Na = 11
        self.Z_I = 53
        self.A_Na = 22.99
        self.A_I = 126.95
        self.rho_NaI = 3.67
        self.M_NaI = 149.89

        n_molecules = (self.rho_NaI / self.M_NaI) * self.NA
        self.N_Na = n_molecules
        self.N_I = n_molecules

        print("Введите параметры детектора :")
        self.R = float(input("Радиус цилиндра R (см): "))
        self.D = float(input("Высота цилиндра D (см): "))
        self.d = self.D / 2

        print("\nВведите координаты источника:")
        self.XO = float(input("XO (см): "))
        self.YO = float(input("YO (см): "))
        self.ZO = float(input("ZO (см): "))

        # Увеличиваем дефолтное значение для лучшей статистики слабого пика
        default_events = 500000
        self.N_events = int(input(f"\nКоличество событий (рекомендуется > {default_events}): ") or default_events)

        self.E_min = 0.0
        self.E_max = 1.5
        self.num_channels = 1024
        self.Cch = None
        self.spectrum = None
        self.source_energy = None
        self.source_name = None
        self._init_planes()

    @staticmethod
    def ray():
        """Генерация случайного изотропного направления"""
        while True:
            l = random.uniform(-1, 1)
            m = random.uniform(-1, 1)
            n = random.uniform(-1, 1)
            length_sq = l ** 2 + m ** 2 + n ** 2
            if length_sq <= 1:
                break
        length = math.sqrt(length_sq)
        return (l / length, m / length, n / length)

    def _init_planes(self):
        """Инициализация плоскостей верхнего и нижнего торцов цилиндра"""
        P1_top = (0, 0, self.d)
        P2_top = (self.R, 0, self.d)
        P3_top = (0, self.R, self.d)
        self.F_top = self.flateABCD(P1_top, P2_top, P3_top)

        P1_bottom = (0, 0, -self.d)
        P2_bottom = (self.R, 0, -self.d)
        P3_bottom = (0, self.R, -self.d)
        self.F_bottom = self.flateABCD(P1_bottom, P2_bottom, P3_bottom)

        self.Ps = (self.XO, self.YO, self.ZO)

    def flateABCD(self, P1, P2, P3):
        """Вычисление коэффициентов плоскости по трем точкам"""
        x1, y1, z1 = P1
        x2, y2, z2 = P2
        x3, y3, z3 = P3

        A = (y2 - y1) * (z3 - z1) - (z2 - z1) * (y3 - y1)
        B = (z2 - z1) * (x3 - x1) - (x2 - x1) * (z3 - z1)
        C = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
        D = -(A * x1 + B * y1 + C * z1)
        return A, B, C, D

    def crossFlat(self, RAY, Ps, F):
        """Нахождение точки пересечения луча с плоскостью"""
        l, m, n = RAY
        x0, y0, z0 = Ps
        A, B, C, D = F

        denominator = A * l + B * m + C * n
        if abs(denominator) < 1e-12:
            return None, None

        t = -(A * x0 + B * y0 + C * z0 + D) / denominator

        if t < 0:
            return None, None

        x = x0 + l * t
        y = y0 + m * t
        z = z0 + n * t
        return t, (x, y, z)

    def insideFlat(self, R_cyl, P_cross):
        """Проверка, находится ли точка пересечения в пределах торца цилиндра"""
        x, y, z = P_cross
        r = math.sqrt(x ** 2 + y ** 2)
        return r <= R_cyl

    def crossCil(self, RAY):
        """Нахождение пересечения луча с бесконечным цилиндром x² + y² = R²"""
        l, m, n = RAY
        x0, y0, z0 = self.Ps

        a = l ** 2 + m ** 2
        if a < 1e-12:
            return None, None

        b = 2 * (x0 * l + y0 * m)
        c = x0 ** 2 + y0 ** 2 - self.R ** 2

        D = b ** 2 - 4 * a * c

        if D < 0:
            return None, None

        t1 = (-b - math.sqrt(D)) / (2 * a)
        t2 = (-b + math.sqrt(D)) / (2 * a)

        t_min = None
        for t in [t1, t2]:
            if t > 1e-12:
                if t_min is None or t < t_min:
                    t_min = t

        if t_min is None:
            return None, None

        x = x0 + l * t_min
        y = y0 + m * t_min
        z = z0 + n * t_min
        return t_min, (x, y, z)

    def insideCil(self, P):
        """Проверка, находится ли точка внутри конечного цилиндра"""
        x, y, z = P
        if abs(z) > self.d:
            return False
        if math.sqrt(x ** 2 + y ** 2) > self.R:
            return False
        return True

    def sigmaPh(self, E, Z):
        """Микроскопическое сечение фотоэффекта на K-оболочке (см²)"""
        if E <= 0:
            return 0
        try:
            # Формула приближенная, но достаточная для MC
            return 6.651e-25 * 4 * math.sqrt(2) * (Z ** 5) / (137 ** 4) * (self.mc2 / E) ** (7 / 2)
        except:
            return 0

    def sigmaK(self, E, Z):
        """Микроскопическое сечение комптон-эффекта (формула Клейна-Нишины) (см²)"""
        if E <= 0:
            return 0
        gamma = E / self.mc2
        if gamma <= 0:
            return 0

        try:
            term1 = (1 - (2 * (gamma + 1) / gamma ** 2)) * math.log(2 * gamma + 1)
            term2 = 1 / 2 + 4 / gamma - 1 / (2 * (2 * gamma + 1) ** 2)
            return 6.651e-25 * (3 * Z) / (8 * gamma) * (term1 + term2)
        except:
            return 0

    def Sigma(self, sigma_ph_Na, sigma_ph_I, sigma_k_Na, sigma_k_I):
        """Вычисление макроскопических сечений (см⁻¹)"""
        Sigma_ph_Na = self.N_Na * sigma_ph_Na
        Sigma_ph_I = self.N_I * sigma_ph_I
        Sigma_ph_total = Sigma_ph_Na + Sigma_ph_I

        Sigma_k_Na = self.N_Na * sigma_k_Na
        Sigma_k_I = self.N_I * sigma_k_I
        Sigma_k_total = Sigma_k_Na + Sigma_k_I

        Sigma_total = Sigma_ph_total + Sigma_k_total

        return Sigma_ph_total, Sigma_k_total, Sigma_total

    def Length(self, Sigma_total):
        """Розыгрыш длины свободного пробега"""
        if Sigma_total <= 0:
            return float('inf')
        return -math.log(random.random()) / Sigma_total

    def Interaction(self, P_cross, l, m, n, L):
        """Вычисление координат точки взаимодействия"""
        x, y, z = P_cross
        x_int = x + l * L
        y_int = y + m * L
        z_int = z + n * L
        return (x_int, y_int, z_int)

    def cost(self, l, m, n, ll, mm, nn):
        """Косинус угла между двумя векторами"""
        return l * ll + m * mm + n * nn

    def Eloss(self, cos_theta, E):
        """Потеря энергии при комптоновском рассеянии (МэВ)"""
        gamma = E / self.mc2
        E_prime = E / (1 + gamma * (1 - cos_theta))
        return E - E_prime

    def Lottery(self, Sigma_ph, Sigma_k, Sigma_total):
        """Выбор типа взаимодействия"""
        if Sigma_total <= 0:
            return None
        rand = random.random()
        p_ph = Sigma_ph / Sigma_total
        return 'ph' if rand < p_ph else 'k'

    def find_entry_point(self, ray_direction):
        """Нахождение точки входа фотона в детектор"""
        l, m, n = ray_direction
        intersections = []

        t_top, P_top = self.crossFlat(ray_direction, self.Ps, self.F_top)
        if t_top is not None and self.insideFlat(self.R, P_top):
            intersections.append((t_top, P_top))

        t_bottom, P_bottom = self.crossFlat(ray_direction, self.Ps, self.F_bottom)
        if t_bottom is not None and self.insideFlat(self.R, P_bottom):
            intersections.append((t_bottom, P_bottom))

        t_cyl, P_cyl = self.crossCil(ray_direction)
        if t_cyl is not None and self.insideCil(P_cyl):
            intersections.append((t_cyl, P_cyl))

        if not intersections:
            return None

        t_min, P_entry = min(intersections, key=lambda x: x[0])
        return P_entry

    def set_source(self, source_type):
        """Установка типа источника"""
        sources = {
            'Am-241': 0.0595,
            'Cs-137': 0.662,
            'Mn-54': 0.835,
            'Na-22': 0.511,
            'Co-60': 1.173,
            'Zn-65': 1.115
        }

        if source_type in sources:
            self.source_energy = sources[source_type]
            self.source_name = source_type
        else:
            print(f"Источник {source_type} не найден. Используется Cs-137.")
            self.source_energy = 0.662
            self.source_name = "Cs-137"

        if source_type == 'Am-241':
            self.E_min = 0.0
            self.E_max = 0.15
        else:
            self.E_min = 0.0
            self.E_max = min(1.5, self.source_energy * 1.5)

        self.Cch = (self.E_max - self.E_min) / self.num_channels
        self.spectrum = [0] * self.num_channels

    def simulate_generic(self, energy_MeV):
        """
        Базовая симуляция одного фотона заданной энергии.
        ИЗМЕНЕНО: Добавлено гауссово размытие для соответствия второму коду.
        """

        # === ЕДИНСТВЕННОЕ ИЗМЕНЕНИЕ: Локальная функция размытия ===
        def register(E_dep):
            """Регистрация события с учетом разрешения детектора (как во 2-м коде)"""
            if E_dep <= self.E_min or E_dep >= self.E_max:
                return
            k_res = 0.03  # Параметр разрешения NaI(Tl)
            sigma_E = k_res * math.sqrt(E_dep)
            E_meas = random.gauss(E_dep, sigma_E)
            if E_meas < self.E_min or E_meas >= self.E_max:
                return
            ch = int((E_meas - self.E_min) / self.Cch)
            if 0 <= ch < self.num_channels:
                self.spectrum[ch] += 1

        # =========================================================

        E = energy_MeV
        l, m, n = self.ray()
        P_entry = self.find_entry_point((l, m, n))

        if P_entry is None:
            return False

        current_point = P_entry

        while E > 0:
            sigma_ph_Na = self.sigmaPh(E, self.Z_Na)
            sigma_ph_I = self.sigmaPh(E, self.Z_I)
            sigma_k_Na = self.sigmaK(E, self.Z_Na)
            sigma_k_I = self.sigmaK(E, self.Z_I)

            Sigma_ph, Sigma_k, Sigma_total = self.Sigma(
                sigma_ph_Na, sigma_ph_I, sigma_k_Na, sigma_k_I
            )

            if Sigma_total <= 0:
                break

            L = self.Length(Sigma_total)
            P_int = self.Interaction(current_point, l, m, n, L)

            if not self.insideCil(P_int):
                break

            interaction_type = self.Lottery(Sigma_ph, Sigma_k, Sigma_total)

            if interaction_type == 'ph':
                # ЗАМЕНА: Было прямое вычисление канала, стало размытие
                register(E)
                return True

            elif interaction_type == 'k':
                l_new, m_new, n_new = self.ray()
                cos_theta = self.cost(l, m, n, l_new, m_new, n_new)
                dE = self.Eloss(cos_theta, E)

                if dE > 0:
                    # ЗАМЕНА: Было прямое вычисление канала, стало размытие
                    register(dE)

                E = E - dE
                if E <= 0:
                    break

                l, m, n = l_new, m_new, n_new
                current_point = P_int

        return False

    def simulate_single_energy(self):
        """
        Симуляция для источника с одной энергией гамма-излучения
        (все источники кроме Cs-137, у которого есть дополнительный рентгеновский пик)
        """
        if self.source_name is None or self.source_energy is None:
            print("Ошибка: источник не установлен!")
            return

        # ВАЖНО: Пересоздаем спектр с правильными параметрами
        self.Cch = (self.E_max - self.E_min) / self.num_channels
        self.spectrum = [0] * self.num_channels

        print(f"Запуск симуляции {self.source_name} ({self.N_events} событий)...")
        print(f"Энергия источника: {self.source_energy} МэВ")
        print(f"Диапазон энергий: {self.E_min} - {self.E_max} МэВ")
        print(f"Ширина канала: {self.Cch:.6f} МэВ")

        registered = 0
        for event in range(self.N_events):
            if event % 50000 == 0:
                print(f"Обработано событий: {event}/{self.N_events}")

            # Для источников с одной энергией просто используем их энергию
            energy = self.source_energy

            if self.simulate_generic(energy):
                registered += 1

        print(f"Симуляция завершена. Зарегистрировано событий: {registered}")

    def simulate_cs137_full(self):
        """
        Специальная симуляция для Cs-137, учитывающая оба канала распада:
        1. Гамма-линия 661.7 кэВ (вероятность ~94.6%)
        2. Рентгеновская линия Ba K-alpha ~32 кэВ (вероятность ~5.4%)
        """
        self.source_name = "Cs-137"
        # Диапазон должен покрывать и 32 кэВ, и 662 кэВ
        self.E_min = 0.0
        self.E_max = 1.0
        self.Cch = (self.E_max - self.E_min) / self.num_channels
        self.spectrum = [0] * self.num_channels

        # Вероятности распада Cs-137
        prob_gamma = 0.946  # 661.7 keV
        prob_xray = 0.054  # ~32 keV (K-shell conversion X-rays)

        # print(f"Запуск полной симуляции Cs-137 ({self.N_events} событий)...")
        # print(f"Ожидаемые пики: 662 кэВ ({prob_gamma * 100:.1f}%) и 32 кэВ ({prob_xray * 100:.1f}%)")

        registered = 0
        for event in range(self.N_events):
            if event % 50000 == 0:
                print(f"Обработано событий: {event}/{self.N_events}")

            # Розыгрыш типа распада
            if random.random() < prob_gamma:
                energy = 0.6617
            else:
                energy = 0.032

            if self.simulate_generic(energy):
                registered += 1

        print(f"Симуляция завершена. Зарегистрировано событий: {registered}")

    def plot_spectrum(self):
        """Построение графика энергетического спектра"""
        plt.figure(figsize=(12, 8))

        energies = [self.E_min + i * self.Cch for i in range(self.num_channels)]

        # ИСПОЛЬЗУЕМ 'steps-post' ДЛЯ ИДЕАЛЬНОГО НУЛЯ
        # Обычный plot соединяет точки линиями, создавая визуальный шум.
        # Гистограмма (steps) показывает реальные дискретные отсчеты.
        plt.step(energies, self.spectrum, 'b-', where='post', linewidth=1.0, alpha=0.8)

        plt.xlabel('Энергия (МэВ)', fontsize=12)
        plt.ylabel('Количество отсчетов', fontsize=12)
        plt.title(f'Энергетический спектр {self.source_name} в детекторе NaI', fontsize=14)
        plt.grid(True, alpha=0.3)

        # Ограничиваем ось Y снизу нулем, чтобы не было видно отрицательных значений (если вдруг)
        plt.ylim(bottom=0)

        plt.tight_layout()
        plt.show()


def main():
    detector = GammaInteraction()

    print("Доступные источники:")
    # print("1. Am-241 - 59.5 кэВ")
    print("1. Cs-137 - 661.7 кэВ")
    print("2. Mn-54 - 835 кэВ")
    print("3. Na-22 - 511 кэВ")
    print("4. Co-60 - 1173 кэВ")
    print("5. Zn-65 - 1115 кэВ")

    choice = input("\nВведите номер источника (1-5) или нажмите Enter для Cs-137: ").strip()

    sources_map = {
        #"1": "Am-241",
        "1": "Cs-137",
        "2": "Mn-54",
        "3": "Na-22",
        "4": "Co-60",
        "5": "Zn-65"
    }

    source_type = sources_map.get(choice, "Cs-137")

    if source_type == "Cs-137":
        detector.simulate_cs137_full()
    else:
        # Для простых источников можно использовать существующий метод
        detector.set_source(source_type)
        # Нужно добавить метод для симуляции одиночного источника
        detector.simulate_single_energy()

    detector.plot_spectrum()


if __name__ == "__main__":
    main()