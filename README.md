# Deeplearning Case 2 NLP
## Team 12

Структура Проекта

```
├── README.md          <- Описание проекта.
├── data
│   ├── processed      <- Обработанные датасеты используемые для обучения модели.
│   └── raw            <- Первоначально подобранные датасеты.
│
├── demo                <- Папка с кодом и скриптами используемых для работы demo_application.
│    │
│    ├── data           <- Сконфигурированное хранилище для работы demo_application.
│    │
│    ├── pages          <- Код страниц demo_application.
│    │
│    └── models         <- Скрипты для работы demo_application.
│
├── notebooks          <- Ноутбуки в которых производилась обработка данных и сравнение моделей для использования при решении кейса.
│
├── poetry.lock        <- Файл с зависимостями для виртуальной среды poetry.
│
└── pyproject.toml     <- Данные для установки пакетов виртуальной среды poetry.
```

Для обучения и работы модели были использованы следующие датасеты:
- датасет с резюме Head Hunter, предоставляемый в рамках кейса: ([HeadHunter_CV]https://drive.google.com/file/d/1ikA_Ht45fXD2w5dWZ9sGTSRl-UNeCVub/view)
- датасет с ваканссиями Head Hunter ([HeadHunter_vacancy]https://www.kaggle.com/datasets/antonbelyaevd/headhunter-vacancies-for-data-search) by Anton Belyaev ([contacts]https://t.me/Suiseki_desu)

# Сравнение методов matching'a 

## spaCy

### Преимущества

- **Высокая производительность:** spaCy обеспечивает быструю обработку текста благодаря оптимизированному коду и эффективным структурам данных.
- **Широкий функционал:** Предоставляет широкий набор инструментов для обработки текста, включая разбор предложений, извлечение именованных сущностей и морфологический анализ.

### Недостатки

- **Ограниченные языковые модели:** Используемые языковые модели обладают низкой точностью для русского языка по сравнению с английским.

## TF-IDF (Term Frequency-Inverse Document Frequency)

### Преимущества

- **Простота:** Крайне простой инструмент в реализации и понимании.
- **Интерпретируемость:** Предоставляет возможность легко интерпретировать веса слов в документе.

### Недостатки

- **Ограниченность:** TF-IDF ориентирована на статистические характеристики, что делает её менее эффективной для понимания семантического содержания текста.
- **Missing context:** Не учитывает контекст и взаимосвязь между словами. 

## RUbert-Tiny2 

### Преимущества

- **Понимание контекста:** RUbert-Tiny2, основанный на BERT, учитывает контекст и семантику в тексте, что делает модель более подходящей для понимания естественного языка.
- **Эффективность:** Несмотря на свою компактность, RUbert-Tiny2 обладает высокой точностью в задачах обработки русского текста. Это делает ее идеальным выбором для проектов, где важны как результаты, так и эффективное использование вычислительных мощностей.
- **Легкость модели:** RUbert-Tiny2 обеспечивает высокую производительность при минимальных требованиях к вычислительным ресурсам. Это особенно важно для задач, где ограничены ресурсы, такие как мобильные устройства или встраиваемые системы.

### Недостатки

- **Ограниченность ресурсов:** В более сложных и масштабных проектах можно столкнуться с ограниченностью ресурсов модели в виду того, что она не сможет обеспечивать такую же глубину представления.
- **Низкое разрешение вопросов семантики:** В сравнении с более крупными моделями, RUbert-Tiny2 может иметь ограиченные возможности в раскрытии сложных семантических взаимосвязей в тексте. 

# Почему RUbert-Tiny2?

RUbert-Tiny2 представляет собой легкую и эффективную модель для обработки русского текста. Эта модель основана на технологии BERT и обучена на разнообразных корпусах данных на русском языке. RUbert-Tiny2 предназначена для задач обработки естественного языка, включая задачи классификации, извлечения информации, и многие другие.

- **Экономия ресурсов:** RUbert-Tiny2 предоставляет возможность эффективной обработки русскоязычных текстов при минимальных затратах на вычислительные ресурсы.
- **Простота внедрения:** Модель разработана с учетом легкости внедрения в различные проекты, обеспечивая быструю интеграцию и запуск ваших приложений на русском языке.
- **Высокая производительность:** Несмотря на свой небольшой размер, RUbert-Tiny2 сохраняет высокую точность и производительность в различных задачах обработки текста. Однако в более масштабных проектах, как упоминалось ранее, может не хватать её ресурсов. В нашем случае модели хватает для решения данного кейса.

Перечисленные ранее недостатки не являются фундаментальными ограничениями, а скорее представляют собой компромиссы, сделанные в пользу легкости и эффективности модели. В большинстве общих случаев RUbert-Tiny2 остается лучшим выбором благодаря своей оптимальной комбинации производительности и точности на русском языке.

# Дальнейшее развитие проекта

В рамках поставленной задачи по второму кейсу выше был представлен инструмент, который справляется с поставленной задачей, однако у нашей команды есть дальнейшее виденье относительно развития данного проекта - **Telegram Bot.** 

Первоначально в рамках работы в команде было выработано решение по запуску модели в рамках **Telegram Bot'a**, который должен был выступать помощником для соискателей в подборе вакансий, и для работодателей в подборе лучших кандидатов по резюме. 

В рамках дальнейшей работы над проектом планируется запустить полноценного **Telegram Bot'a**, который позволит реализовать:
- Распознавание и сопоставление резюме и вакансий. 
- Советы по улучшению резюме или вакансии.
- Диалоговый формат, в котором можно будет оценивать резюме и получать контакты прямо в чате, что поможет быстрее наладить коммуникационные связи между работодателем и соискателем, в случае если их интересы совпадут.

([пример работы Telegram Bot]case2cv\case2cv\Telegram Bot.png)

По мере продвижения разработки **Telegram Bot'a** и развития данного проекта будут выкладываться обновления в отдельный репозиторий, ссылка на который появится позднее. 

### Developing team

- **Артемий Зитнер**
- **Михаил Иванов**
- **Ван Цюаньюй**
- **Renzo A**
