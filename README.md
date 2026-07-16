# Desafio MBA — Ingestão e Busca com IA

Breve projeto para importar documentos, indexá-los e expor uma interface de busca/chat baseada em embeddings.

## Requisitos
- Python 3.8+ instalado
- Docker & Docker Compose (para serviços dependentes)

## Instalação
1. Crie e ative um ambiente virtual:

```bash
python3 -m venv venv
source venv/bin/activate
```

2. Instale dependências:

```bash
pip install -r requirements.txt
```

## Como executar
1. Inicie serviços dependentes (ex.: banco de dados):

```bash
docker compose up -d
```

2. Ingestão de documentos (ex.: PDF). Ajuste caminhos/credenciais dentro de `src/ingest.py` se necessário:

```bash
python src/ingest.py
```

3. Inicie a interface de chat/busca:

```bash
python src/chat.py
```

Observação: o repositório também contém `src/search.py` para chamadas diretas de busca/consulta.

## Estrutura do repositório
- `src/ingest.py` — lida com leitura e indexação de documentos
- `src/search.py` — funções/CLI para consultar o índice
- `src/chat.py` — interface de chat que utiliza a camada de busca
- `requirements.txt` — dependências Python
- `docker-compose.yml` — serviços auxiliares (ex.: banco, vector DB)

## Dicas
- Verifique as variáveis de ambiente ou as configurações dentro dos arquivos em `src/` antes de rodar.
- Use ambientes virtuais para manter dependências isoladas.

## Contato
Se precisar, abra uma issue ou me mande uma mensagem no repositório.
