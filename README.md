# Desafio MBA — Ingestão e Busca com IA

Projeto RAG (Retrieval-Augmented Generation) para importar documentos PDF, indexá-los via embeddings, e expor uma interface de chat/busca baseada em contexto.

## Requisitos
- Python 3.8+ instalado
- Docker & Docker Compose (para PostgreSQL + pgvector)
- Credenciais de API: Google Generative AI ou OpenAI (pelo menos uma)

## Instalação

### 1. Ambiente virtual
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Dependências
```bash
pip install -r requirements.txt
```

### 3. Configuração de variáveis de ambiente

Copie `.env.example` para `.env` e preencha com suas credenciais:

```bash
cp .env.example .env
```

Variáveis obrigatórias:
- `PGVECTOR_URL` — string de conexão PostgreSQL com pgvector (ex: `postgresql+psycopg://postgres:postgres@localhost:5432/rag`)
- `PGVECTOR_COLLECTION` — nome da coleção de vetores (ex: `documents`)
- `PDF_PATH` — caminho relativo ou absoluto do PDF a ingerir (ex: `./document.pdf`)

Escolha **uma** provedora de embeddings e LLM:
- **Google**: defina `GOOGLE_API_KEY`
- **OpenAI**: defina `OPENAI_API_KEY`

Opcional:
- `TOP_K` — número de documentos a recuperar (padrão: 4)
- `GOOGLE_EMBEDDING_MODEL`, `GOOGLE_LLM_MODEL`, `OPENAI_EMBEDDING_MODEL`, `OPENAI_LLM_MODEL`

## Como executar

### Inicie o banco de dados
```bash
docker compose up -d
```

### Ingestão de PDF
```bash
python src/ingest.py
```

**Opções de CLI:**
```bash
python src/ingest.py --pdf-path ./outro_documento.pdf --collection outra_colecao --overwrite
```

- `--pdf-path` — caminho alternativo para o PDF
- `--collection` — nome alternativo da coleção
- `--overwrite` — limpar a coleção antes de ingerir novos documentos

### Chat/Busca interativa
```bash
python src/chat.py
```

Digite suas perguntas; responda "sair" para encerrar.

### Busca direta (sem chat)
```bash
python -c "from src.search import search_prompt; print(search_prompt('sua pergunta'))"
```

## Estrutura do código

```
src/
  __init__.py           — marcador de pacote Python
  ingest.py             — carregamento, chunking, indexação de PDFs
  search.py             — retriever, chain RAG e geração de respostas
  chat.py               — interface interativa de chat
config.py               — configurações centralizadas (BaseSettings com Pydantic)
docker-compose.yml      — PostgreSQL + pgvector
requirements.txt        — dependências Python
.env                    — credenciais e variáveis (não commitado)
.env.example            — modelo de variáveis de ambiente
```

## Arquitetura

```
PDF → Loader → Chunks (1000 tokens) → Embeddings (Google/OpenAI)
                                         ↓
                                    PGVector (PostgreSQL)
                                         ↓
                                    Retriever (top-k)
                                         ↓
Pergunta → Embeddings → Vector Search → LLM (com contexto) → Resposta
```

## Configuração de provedores

### Google Generative AI
```env
GOOGLE_API_KEY=sua_chave_aqui
GOOGLE_EMBEDDING_MODEL=models/embedding-001
GOOGLE_LLM_MODEL=gemini-pro
```

### OpenAI
```env
OPENAI_API_KEY=sua_chave_aqui
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_LLM_MODEL=gpt-4o-mini
```

## Troubleshooting

**Erro: "No embedding provider configured"**
- Certifique-se de ter definido `GOOGLE_API_KEY` ou `OPENAI_API_KEY` em `.env`

**Erro: "PDF file not found"**
- Verifique se `PDF_PATH` em `.env` aponta para um arquivo existente

**Erro: "Connection refused" (banco de dados)**
- Execute `docker compose up -d` para iniciar PostgreSQL

**Erro: "No documents were generated"**
- O PDF pode estar vazio ou apenas contém imagens sem texto extraível

## Estrutura de projeto

- `src/ingest.py` — funções: `load_pdf()`, `split_documents()`, `build_embedding_model()`, `ingest_pdf()`
- `src/search.py` — funções: `build_retriever()`, `build_llm()`, `search_prompt()`
- `src/chat.py` — função: `main()` para loop interativo
- `config.py` — classe `Settings` para validação de ambiente

## Dicas de desenvolvimento

- Use `--overwrite` ao alterar significativamente o prompt ou embeddings para reindexar
- Leia os logs para entender o fluxo de chunking e embeddings
- Para debug, altere o nível de logging em `config.py` ou nos módulos
- O timeout padrão para LLMs é 60s; ajuste em caso de APIs lentas

## Contato

Abra uma issue ou envie um pull request no repositório.

