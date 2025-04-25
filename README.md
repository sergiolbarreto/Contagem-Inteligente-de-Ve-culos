# 📘 Sistema Inteligente de Contagem de Carros em um Semáforo

## **Introdução** 📜

O crescimento do tráfego urbano exige soluções tecnológicas que otimizem a mobilidade e melhorem a gestão do trânsito nas grandes cidades. Neste contexto, desenvolveu-se um sistema inteligente para contagem de veículos em semáforos a partir de vídeos, eliminando a necessidade de sensores físicos e permitindo maior flexibilidade e economia na implantação.

Este sistema é capaz de identificar, rastrear e contar veículos de forma precisa, respeitando o estado do semáforo (verde ou vermelho), e evitando duplicidades na contagem. Ao final do processamento, são gerados relatórios com estatísticas relevantes que podem ser utilizados por órgãos de trânsito para otimizar o tempo de abertura e fechamento dos sinais.

## **Objetivo** 🎯

O principal objetivo do projeto é criar um sistema automatizado que:

- Detecta veículos em vídeos de tráfego urbano.
- Reconhece o estado do semáforo (verde ou vermelho).
- Conta os veículos que cruzam uma linha virtual apenas quando o semáforo está verde.
- Armazena os dados em um arquivo estruturado (CSV).
- Gera um relatório com as estatísticas de contagem e influência do semáforo no fluxo.

## **Tecnologias e Bibliotecas Utilizadas** 🛠️

- **Python**: Linguagem de programação principal.
- **OpenCV**: Processamento de vídeo, leitura e escrita de arquivos .mp4, manipulação de frames e conversão de cores.
- **YOLOv8 (Ultralytics)**: Modelo de detecção de objetos para identificar veículos e semáforos.
- **NumPy**: Operações com arrays e manipulação numérica.
- **Pandas**: Manipulação de dados e geração de arquivos CSV.
- **Matplotlib**: Geração de gráficos e visualizações.
- **SciPy (distance.cdist)**: Cálculo de distâncias entre centroides para rastreamento.

## **Justificativa da Escolha do Modelo** 🤖

Optei pelo modelo **YOLOv8** (You Only Look Once, versão 8) devido à sua alta precisão e velocidade em tarefas de detecção em tempo real. A versão "n" (nano) foi minha escolha para equilibrar desempenho e tempo de processamento, possibilitando análises mais rápidas mesmo em máquinas com recursos limitados.

Além disso, escolhi o YOLOv8 porque ele já vem pré-treinado com um grande conjunto de classes, incluindo veículos e semáforos, o que facilita sua aplicação direta no problema.

## **Vídeos Utilizados** 🎥

Foram utilizados dois vídeos para testar a aplicação e validar o comportamento do sistema em diferentes situações:

- **Vídeo 1**: Apresenta apenas o semáforo na cor verde durante toda a duração. Esse vídeo foi utilizado para validar a contagem contínua de veículos sem interrupções.
  ![image (10)](https://github.com/user-attachments/assets/ec3fe657-ec40-434b-b490-5a5fae682053)

- **Vídeo 2**: Inicia com o semáforo verde e, em determinado momento, muda para vermelho. Esse caso permite testar a capacidade do sistema de pausar a contagem corretamente enquanto o sinal está fechado.
![image (11)](https://github.com/user-attachments/assets/d3c26131-9a58-40d4-a387-2cb95bced17f)


Ambos os vídeos foram obtidos de bancos de vídeos públicos gratuitos:

- Um foi retirado da **Freepik** ([https://www.freepik.com/](https://www.freepik.com/)).
- O outro foi obtido na **Videvo** ([https://www.videvo.net/](https://www.videvo.net/)).

## **Arquitetura e Funcionamento do Sistema** 🏗️

### **Captura do Vídeo**

- O sistema permite que o usuário forneça um vídeo de tráfego urbano.
- Cada frame do vídeo é processado individualmente.

### **Detecção de Veículos e Semáforo**

- O YOLOv8 detecta veículos (carros, caminhões, ônibus e motos) e também localiza a área do semáforo.
- Após detectar o semáforo, a região é convertida para HSV para avaliar se a cor predominante é verde ou vermelha.

### **Rastreamento e Contagem**

- Utiliza-se um **rastreador de centróides** baseado em distância Euclidiana.
- Uma **linha virtual** é posicionada no vídeo. Quando o centro do veículo cruza essa linha e o semáforo está **verde**, o veículo é contado.
- Um sistema de memória garante que cada veículo seja contado **apenas uma vez**.

### **Armazenamento dos Dados**

- As informações de cada contagem (ID do veículo, tempo, frame, estado do semáforo) são armazenadas em uma lista.
- Ao final do processamento, os dados são salvos em um arquivo **CSV** para futura análise.

## **Resultados**

Os resultados obtidos foram bem satisfatórios. O modelo perfomou bem nos dois casos e foi capaz de fazer a contagem dos veículos sem repetição e sempre levando em consideração o semáforo. Quando esse estava vermelho, os veículos não eram contados.

Ao final do processamento de um vídeo, o sistema gera:

- A contagem total de veículos.
- A distribuição de veículos ao longo do tempo.
- A influência do estado do semáforo no fluxo (ex: maior volume com sinal verde, menor durante o vermelho).
- Um vídeo de saída com os objetos detectados, IDs, linha de contagem e estado atual do semáforo visivelmente destacados.

## **Conclusão** ✅

O sistema desenvolvido demonstrou ser eficaz na detecção, rastreamento e contagem de veículos em vídeos urbanos, mesmo com a presença de semáforos. A integração do YOLOv8 com o OpenCV permitiu uma solução flexível, escalável e sem necessidade de sensores físicos.

A abordagem baseada em visão computacional oferece uma alternativa poderosa para sistemas de monitoramento de tráfego, possibilitando decisões mais inteligentes e baseadas em dados reais.

## **Próximos Passos e Melhorias Futuras** 🚀

- Implementar dashboard interativo com gráficos para análise dos dados.
- Adicionar classificação por tipo de veículo (carro, ônibus, moto, caminhão).
- Testar com diferentes ângulos de câmera e condições de iluminação.
- Suporte à análise em tempo real.
