# System do wykrywania arytmii serca na podstawie danych opisujących pacjenta

Zadanie polegało na przetworzeniu zbioru danych Arrhythmia Data Set pochodzącego z UCI Machine Learning Repository (https://archive.ics.uci.edu/ml/datasets/arrhythmia), a następnie wytrenowaniu na jego podstawie modelu AI oraz utworzenia drzewa decyzyjnego.

## Wyniki
Gęsta sieć neuronowa osiągała skuteczność ok. 60-70%, metoda wykorzystująca drzewa decyzyjne była minimalnie lepsza i osiągała skuteczność rzędu 65-75%.

## Wnioski
Dane dostępne w UCI Machine Learning Repository zawierają 15 różnych rodzajów klasyfikacji, w moim projekcie ograniczyłem się do utworzenia klasyfikatora binarnego, który wykrywał jedynie obecność arytmii, a nie jej dokładny rodzaj.

Niektóre rodzaje były słabo reprezentowane w zbiorze treningowym (pojedyncze wystąpienia), co mocno zmniejszyło skuteczność modelu w ich wykrywaniu.

Największym problemem w projekcie była niewystarczająca ilość danych treningowych (ok. 450 rekordów).

## Wykorzystane technologie:
- tensorflow
- keras
- sklearn
- numpy
- matplotlib

Projekt był realizowany w ramach przedmiotu Sztuczna Inteligencja
