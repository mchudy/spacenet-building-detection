# spacenet-building-detection

Instalacja zależności:
```
make requirements
```

Pobranie danych (potrzebne jest konto na AWSie i skonfigurowanie AWS CLI - https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-getting-started.html):
```
make download-data city={miasto}
```
Dostępne miasta: `rio`, `vegas`, `paris`, `paris2`, `shanghai`, `khaortum`.
