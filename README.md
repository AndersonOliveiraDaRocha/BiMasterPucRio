# Visão Computacional para Detecção de Corrosão Externa em Instalações Offshore

* BI MASTER PUC RIO (2020 - TURMA PETROBRAS) GRUPO D5 DO PROGRAMA E&P COMPETÊNCIAS.
* ALUNO: ANDERSON OLIVEIRA DA ROCHA
* MATRÍCULA: 192190080

## Contexto: 

Instalações industriais offshore operam em ambientes com alta exposição a intempéries, nesse ambiente hostil, estruturas metálicas sofrem com a corrosão e acabam por se degradar, comprometendo dessa forma a integridade da unidade e gerando potencial de comprometer a eficiência operacional e a segurança dos trabalhadores. Inspeções periódicas acontecem com o objetivo de identificar ocorrências de corrosão de modo a subsidiar um adequado planejamento e priorização da execução dos serviços de restauração e preservações desses elementos. Contudo, o trabalho de inspecionar periodicamente essas unidades, além é oneroso, consome um HH considerável, além de concorrer por vagas a bordo, um problema constante em uma unidade offshore.

## Proposta:

Considerando o contexto acima descrito, esse projeto tem como objetivo permitir a detecção da corrosão externa por meio de um algoritmo de visão computacional deeplearning que analisa uma fotografia comum e destaca as áreas com corrosão existente. Essa abordagem viabiliza uma triagem em terra com foco em entender as condições gerais da unidade. Essa abordagem tem potencial de reduzir a quantidade de embarques e mesmo a quantidade de HH envolvido no processo de inspeção.

## Abordagem:

Entre as várias técnicas de visão computacional descritas abaixo, a técnica de segmentação por instância foi selecionada para identificação das múltiplas ocorrências de corrosão nas fotos das estruturas metálicas. A técnica de segmentação por instância é uma das técnicas mais avançadas e considerando o problema em questão ela pavimenta a evolução do modelo para um cenário futuro, considerando que no momento só será feita a detecção de ocorrências de corrosão, porém se espera evoluir o modelo para identificar o grau de severidade da corrosão, como por exemplo: leve, moderada ou severa.

### * Classificação de Imagens

Na classificação de imagens o objetivo é identificar a qual classe pertence uma determinada imagem, como carros, pessoas, animais, etc.

<img src="IMAGES/ImgClassification.jpeg" width="1000">


### * Detecção de Objetos

Na detecção de objetos o objetivo é localizar onde os elementos de interesse (uma determinada classe de interesse) se encontram na imagem, geralmente por meio de uma Bounding Box (caixa) ao redor do mesmo. Se combinada com outras técnicas é possível além de localizar e marcar os elementos na imagem com um Bounding Box, também classificar em classes distintas.

<img src="IMAGES/ObjDetection.jpeg" width="1000">


### * Rastreamento de Objetos

Rastreamento de objetos é o processo de seguir um ou mais elementos específicos de interesse em uma cena, geralmente utilizado em aplicações com vídeos.


<img src="IMAGES/ObjTracking.jpeg" width="1000">

### * Segmentação Semântica

Na segmentação semântica um conjunto de pixels que pertence a uma mesma classe são classificados de forma igual formando uma máscara que delimita as fronteiras do elemento.

<img src="IMAGES/SemanticSegm.jpeg" width="1000">

### * Segmentação por Instância

Na segmentação por instância os pixels são classificados por semelhança, não somente das classes, mas também por cada instância que pode ser de classes distintas. Uma máscara é criada individualmente para cada instância identificada na cena.

<img src="IMAGES/InstanceSegm.jpeg" width="1000">

Considerando a técnica selecionada para abordar o problema, as pesquisas desse estudo indicaram a rede MASK RCNN como um caminho promissor para obtenção de resultados. Uma implementação da MASK RCNN pode ser encontrada em: https://github.com/matterport/Mask_RCNN e foi utilizada como base desse estudo.

## Características:

Como citado no tópico anterior esse projeto foi desenvolvido com base na rede MASK RCNN, seguem abaixo algumas características técnicas:

* KERAS com BACKEND Tensorflow;
* Rede MASK RCNN com BACKBONE RESNET101;
* Rede pré-treinada com DATASET MS COCO;
* Retreino com DATASET customizado com anotações de segmentação de corrosão, conjunto de treino (TRAIN) com 100 imagens e conjunto de validação (VAL) com 30 imagens;
* Anotações realizadas com o VGG Image Annotator (VIA) e anotações exportadas como JSON (DATASET.JSON);
* Treino realizado em GPU NVIDIA 940MX com CUDA 9.0 e CUDNN64_7.dll;
* Predição realizada em CPU;
* Requisitos importantes de versão: tensorflow==1.5.1; keras==2.1.0; tensorflow-gpu==1.5.1;h5py==2.10.0;
* Os pesos MS COCO e CORROSION, respectivamente para retreinar o modelo e realizar predições, encontram-se compactados na pasta **WEIGHT** e precisam ser descompactados para execução do projeto;

A estrutura de pastas do projeto encontra-se disposta da seguinte forma:

* Pasta **CUSTOM**: Contém os arquivos **custom.py** e **predict.py**, respectivamente para fazer o treinamento do modelo e a predição com a detecção da corrosão. Para treinar o modelo e gerar novos pesos deve-se usar o seguinte comando: **"python custom.py train --dataset=../dataset --weights=coco"**, os novos pesos gerados estarão na SUBPASTA de sessão localizada na pasta **WEIGHT** na raiz do projeto.  
Para predição, uma imagem simples precisa ser copiada para pasta **CUSTOM** e deve ter o nome: **Corrosao.jpg**, posteriormente deve ser executado o comando **pyhton predict.py**. Ainda na pasta CUSTOM existe uma SUBPASTA chamada RESULTADOS onde algumas predições estão salvas para efeito de demonstração, como segue:


<img src="CUSTOM/Resultados/Sample_A.jpg" width="154"><img src="CUSTOM/Resultados/Result_A.png" width="160">
<img src="CUSTOM/Resultados/Sample_B.jpg" width="154"><img src="CUSTOM/Resultados/Result_B.png" width="160">
<img src="CUSTOM/Resultados/Sample_F.jpg" width="154"><img src="CUSTOM/Resultados/Result_F.png" width="160">

* Pasta **DATASET**: Hospeda os conjuntos de dados utilizados para treinar o modelo, DATASET de Treino (**TRAIN**) e DATASET de validação (**VAL**). As anotações foram feitas no VIA e os resultados exportados no formato JSON como já explicado anteriormente. Para um modelo com maior precisão é necessário um volume de dados de treinamento mais expressivo, porém, não houve disponibilidade para o exercício desse projeto.

* Pasta **mrcnn**: Hospeda o núcleo do modelo MASK RCNN.

* Pasta **WEIGHT**: Estão armazenados os pesos. Primeiro existe o peso **COCO_WEIGHT.h5** que é o peso do DATASET MS COCO, usado como base para estender o projeto, depois os pesos treinados na detecção de corrosão que estão salvos com o nome de **CORROSION.h5**. Caso o modelo seja novamente treinado serão geradas novas pastas de sessão e os novos pesos treinados serão salvos dentro dessas pastas, cada época vai gerar um peso. Importante descompactar os pesos antes da execução do projeto!

## Desenvolvimento:

O desenvolvimento do projeto teve como principais fases:

**1.** Autorização do corpo gerencial da cia. (PETROBRAS) para propor a temática de detecção de corrosão por visão computacional e acesso às fontes de dados;  
**2.** Análise do conteúdo base para o treino do modelo (DATASET);  
**3.** Complementação de anotações e tratamento de imagens (Giro por exemplo);  
**4.** Pesquisa com foco em identificar uma abordagem técnica para segmentação de imagens por instância;  
**5.** Análise do modelo MASK RCNN, entendimento do código fonte, implementações e ajustes necessários para adaptar ao problema de detecção de corrosão;  
**6.** Configuração de ambiente com os pacotes corretos das bibliotecas envolvidas;  
**7.** Ajustes para execução dos processamentos em GPU;  
**8.** Treino do modelo para o novo problema de detecção de corrosão externa;  
**9.** Ajuste de parâmetros (épocas, passos, leaning rate, entre outros) para buscar os melhores resultados nas métricas de loss;  
**10.** Predições com imagem nunca antes vistas pelo modelo;  
**11.** Predições também com imagens envolvidas nos conjuntos de treino e validação;  
**12.** Refactoring eliminando funções desnecessárias ao objetivo proposto;  
**13.** Upload do projeto para o GITHUB;  
**14.** Documentação do projeto;  

## Desenvolvimento Técnico:

As implementações descritas no item 5 do capítulo anterior ocorreram majoritariamente nos arquivos  **custom.py** e **predict.py** onde o código fonte encontra-se comentado para facilitar entendimento e eventuais necessidades de manutenção.  
  
O modelo foi treinado considerando alguns parâmetros:  
  
LEARNING_RATE = 0.001  
EPOCHS=20  
LAYERS='heads' **(limitação por conta da GPU disponível para o projeto)**  
STEPS_PER_EPOCH = 20  
DETECTION_MIN_CONFIDENCE = 0.8  

O modelo também reprojetado para considerar apenas 2 classes: O BACKGROUND (padrão na implementação MASK RCNN) e a CLASSE ALVO, no caso CORROSAO = SIM, conforme anotação ilustrada abaixo no VIA:  

<img src="IMAGES/VIA.png" width="1000">





