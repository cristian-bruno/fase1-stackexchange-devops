#!/usr/bin/env python3
"""
Gera o dataset da Fase 1 (Ciência de Dados e IA) a partir do
Stack Exchange Data Dump (ex.: devops.stackexchange.com).

Entrada : Posts.xml e Users.xml (extraídos do .7z do site escolhido)
Saída   : dataset_stackexchange.csv  -> arquivo para carregar no Orange
          estatisticas_colunas.csv   -> tabela com as estatísticas pedidas no template
          relatorio_recorte.txt      -> passos do recorte (para documentar no trabalho)

Uso:
    pip install pandas
    python gerar_dataset.py --posts Posts.xml --users Users.xml --site devops.stackexchange.com

Fonte dos dados: Stack Exchange Network, licença CC BY-SA 4.0.
"""
import argparse
import re
import xml.etree.ElementTree as ET

import pandas as pd


def ler_xml(caminho, filtro=None):
    """Lê um XML do dump (linhas <row .../>) e devolve lista de dicionários."""
    linhas = []
    for _, elem in ET.iterparse(caminho, events=("end",)):
        if elem.tag == "row":
            if filtro is None or filtro(elem.attrib):
                linhas.append(dict(elem.attrib))
            elem.clear()  # libera memória
    return linhas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--posts", required=True)
    ap.add_argument("--users", required=True)
    ap.add_argument("--site", default="(site não informado)")
    ap.add_argument("--out", default="dataset_stackexchange.csv")
    ap.add_argument("--stats", default="estatisticas_colunas.csv")
    ap.add_argument("--relatorio", default="relatorio_recorte.txt")
    args = ap.parse_args()

    passos = []

    # 1) Perguntas (PostTypeId = 1)
    posts = ler_xml(args.posts, filtro=lambda a: a.get("PostTypeId") == "1")
    df = pd.DataFrame(posts)
    passos.append(f"1. Posts.xml lido; mantidas apenas perguntas (PostTypeId=1): {len(df)} linhas.")

    # 2) Usuários
    users = pd.DataFrame(ler_xml(args.users))
    users = users[["Id", "Reputation", "CreationDate"]].rename(
        columns={"Id": "OwnerUserId", "Reputation": "ReputacaoAutor",
                 "CreationDate": "DataCriacaoConta"})
    passos.append(f"2. Users.xml lido: {len(users)} usuários.")

    # 3) Colunas obrigatórias (sem elas a linha é descartada -> zero faltantes)
    obrig = ["Id", "CreationDate", "Score", "ViewCount", "AnswerCount",
             "CommentCount", "Title", "Body", "Tags", "OwnerUserId"]
    antes = len(df)
    df = df.dropna(subset=obrig)
    passos.append(f"3. Removidas perguntas com campo obrigatório vazio "
                  f"({', '.join(obrig)}): {antes - len(df)} removidas, {len(df)} restantes.")

    # 4) Junta com a tabela de usuários (descarta autores sem registro)
    antes = len(df)
    df = df.merge(users, on="OwnerUserId", how="inner")
    passos.append(f"4. Junção com Users.xml por OwnerUserId; autores sem registro descartados: "
                  f"{antes - len(df)} removidas, {len(df)} restantes.")

    # 5) Tipos e colunas derivadas
    for c in ["Id", "Score", "ViewCount", "AnswerCount", "CommentCount", "ReputacaoAutor"]:
        df[c] = pd.to_numeric(df[c])
    df["CreationDate"] = pd.to_datetime(df["CreationDate"])
    df["DataCriacaoConta"] = pd.to_datetime(df["DataCriacaoConta"])

    df["TituloTamanho"] = df["Title"].str.len()
    df["CorpoTamanho"] = df["Body"].str.len()
    # Tags vêm como "<docker><kubernetes>"
    df["NumTags"] = df["Tags"].apply(lambda t: len(re.findall(r"<[^<>]+>", t)))
    df["TemCodigo"] = df["Body"].str.contains("<code>", regex=False).map({True: "sim", False: "nao"})
    df["HoraCriacao"] = df["CreationDate"].dt.hour
    df["DiaSemana"] = df["CreationDate"].dt.day_name()
    df["AnoCriacao"] = df["CreationDate"].dt.year
    df["IdadeContaDias"] = (df["CreationDate"] - df["DataCriacaoConta"]).dt.days
    df["TemRespostaAceita"] = df["AcceptedAnswerId"].notna().map({True: "sim", False: "nao"}) \
        if "AcceptedAnswerId" in df.columns else "nao"

    passos.append("5. Colunas derivadas criadas: TituloTamanho, CorpoTamanho, NumTags, TemCodigo, "
                  "HoraCriacao, DiaSemana, AnoCriacao, IdadeContaDias; "
                  "coluna alvo TemRespostaAceita (sim se AcceptedAnswerId existe, senão nao).")

    # 6) Seleção final (colunas de texto longo ficam de fora; só entram numéricas/categóricas)
    final = df[["Id", "Score", "ViewCount", "AnswerCount", "CommentCount",
                "TituloTamanho", "CorpoTamanho", "NumTags", "TemCodigo",
                "HoraCriacao", "DiaSemana", "AnoCriacao",
                "ReputacaoAutor", "IdadeContaDias", "TemRespostaAceita"]].copy()
    # IdadeContaDias negativa seria inconsistência -> descarta
    antes = len(final)
    final = final[final["IdadeContaDias"] >= 0]
    passos.append(f"6. Removidas linhas com idade da conta negativa (inconsistência): "
                  f"{antes - len(final)}; total final: {len(final)} linhas e {final.shape[1]} colunas.")

    assert final.isna().sum().sum() == 0, "Ainda há valores faltantes!"
    final.to_csv(args.out, index=False, encoding="utf-8")

    # 7) Estatísticas por coluna (para a seção 3 do template)
    nominais = {"TemCodigo", "DiaSemana", "TemRespostaAceita"}
    registros = []
    for c in final.columns:
        s = final[c]
        numerica = pd.api.types.is_numeric_dtype(s) and c not in nominais
        moda = s.mode()
        reg = {
            "coluna": c,
            "tipo": "Nominal" if c in nominais else ("Numérico" if c != "Id" else "Numérico (identificador)"),
            "distintos": s.nunique(),
            "minimo": s.min(),
            "maximo": s.max(),
            "moda": moda.iloc[0] if len(moda) else "",
            "media": round(s.mean(), 3) if numerica else "",
            "desvio_padrao": round(s.std(), 3) if numerica else "",
            "mediana": s.median() if numerica else "",
        }
        registros.append(reg)
    pd.DataFrame(registros).to_csv(args.stats, index=False, encoding="utf-8")

    # 8) Distribuição da coluna alvo
    dist = final["TemRespostaAceita"].value_counts()
    passos.append("7. Distribuição da coluna alvo TemRespostaAceita: "
                  + ", ".join(f"{k}={v}" for k, v in dist.items()) + ".")

    with open(args.relatorio, "w", encoding="utf-8") as f:
        f.write(f"Recorte do Stack Exchange Data Dump - site: {args.site}\n")
        f.write("Fonte: https://archive.org/details/stackexchange (CC BY-SA 4.0)\n\n")
        f.write("\n".join(passos) + "\n")

    print("\n".join(passos))
    print(f"\nArquivos gerados: {args.out}, {args.stats}, {args.relatorio}")


if __name__ == "__main__":
    main()
