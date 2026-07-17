import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from search import search_prompt
except ImportError:
    from src.search import search_prompt


def main():
    chain = search_prompt()

    if not chain:
        print("Não foi possível iniciar o chat. Verifique os erros de inicialização.")
        return

    print("Chat iniciado. Digite 'sair' para encerrar.")

    while True:
        prompt = input("PERGUNTA DO USUÁRIO: ").strip()
        if prompt.lower() in {"sair", "exit", "quit"}:
            print("Encerrando o chat.")
            break

        try:
            answer = chain.invoke({"query": prompt})
            output_key = "result"
            result = answer[output_key] if isinstance(answer, dict) else answer
            print(f"\nResposta:\n{result}\n")
        except Exception as exc:
            print(f"Erro ao processar a pergunta: {exc}")


if __name__ == "__main__":
    main()
