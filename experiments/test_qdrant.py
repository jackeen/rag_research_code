"""
Qdrant demon
"""

# from typing import Sequence
from operator import index

import ollama
from fastembed import SparseEmbedding, SparseTextEmbedding
from qdrant_client import QdrantClient, models
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    Payload,
    PointIdsList,
    PointStruct,
    VectorParams,
)

import sys_config


def embedding(content: str) -> list[float]:
    res = ollama.embeddings(
        model=sys_config.OLLAMA_GEMMA_EMBEDDING_MODEL_768, prompt=content
    )
    return list(res.embedding)


def sparse_embedding(content: str) -> SparseEmbedding:
    splade = SparseTextEmbedding(model_name=sys_config.SPARES_PP_EN_v1)
    return next(iter(splade.embed([content])))


client = QdrantClient(host=sys_config.QDRANT_HOST, port=sys_config.QDRANT_PORT)

# here proposals need more background information
query = """
proposals are taken through clearly defined stages, from stage 0, which represents the newest proposals, up through stage 4, which represents the finished proposals.
"""

query_2 = """
What are the official stages of the ECMAScript proposal process?
"""

query_11 = """
It’s a cliche at this point to talk about JavaScript Fatigue, but the source of this fake illness can be traced back to the building process.
"""

query_12 = "It’s a cliche at this point to talk about JavaScript Fatigue"

query_ip_address = """Seeing as most of us would have a hard time remembering what IP address is needed"""

query_stragegy = """
Congratulations on your first pattern!
You just applied your first design pattern—the **STRATEGY** Pattern. That's right, you used the Strategy Pattern to rework the SimUDuck app.
Thanks to this pattern, the simulator is ready for any changes those execs might cook up on their next business trip to Maui.
Now that we've made you take the long road to learn it, here's the formal definition of this pattern:
**The Strategy Pattern** defines a family of algorithms, encapsulates each one, and makes them interchangeable. Strategy lets the algorithm vary independently from clients that use it.
Use THIS definition when you need to impress friends and influence key executives.
Below you'll find a mess of classes and interfaces for an action adventure game. You'll find classes for game characters along with classes for weapon behaviors the characters can use in the game. Each character can make use of one weapon at a time, but can change weapons at any time during the game. Your job is to sort it all out...
(Answers are at the end of the chapter.)
"""

query_pattern = (
    """**The Observer Pattern** defines a one-to-many dependency between objects"""
)

query_memory = """When working memory is full"""


def dense_query(q: str):
    res = client.query_points(
        collection_name="temp_f",
        query=embedding(q),
        using="text_dense_vector",
        limit=4,
        score_threshold=0.0,
    )
    for p in res.points:
        print("------------------------------------------")
        if p.payload:
            content = p.payload.get("page_content", "")
            print(p.score)
            print(content.replace("\n", " ")[:500])


def hybrid_query(q: str):
    sparse_emb = sparse_embedding(q)
    res = client.query_points(
        collection_name="p_exp_3_md_multi_layer_pg_6",
        prefetch=[
            models.Prefetch(query=embedding(q), using="text_dense_vector", limit=20),
            models.Prefetch(
                query=models.SparseVector(
                    indices=sparse_emb.indices.tolist(),
                    values=sparse_emb.values.tolist(),
                ),
                using="text_sparse_vector",
                limit=20,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=4,
    )
    for p in res.points:
        print("------------------------------------------")
        if p.payload:
            content = p.payload.get("page_content", "")
            print(p.score)
            print(content.replace("\n", " ")[:500])


if __name__ == "__main__":
    print("\n\n\n")
    # print(SparseTextEmbedding.list_supported_models())
    # dense_query(query)
    #

    hybrid_query(query_memory)

    # x = """
    # Since its release in 1995, JavaScript has gone through many changes. At first, it made adding interactive elements to web pages much simpler. Then it got more robust with DHTML and AJAX. Now, with Node.js, JavaScript has become a language that is used to build full-stack applications. The committee that is and has been in charge of shep‐ herding the changes to JavaScript is the European Computer Manufacturers Associa‐ tion (ECMA).
    # Changes to the language are community-driven. They originate from proposals that community members write. Anyone [can submit a proposal](https://tc39.github.io/process-document/) to the ECMA committee. The responsibility of the ECMA committee is to manage and prioritize these propos‐ als in order to decide what is included in each spec. Proposals are taken through clearly defined stages, from stage 0, which represents the newest proposals, up through stage 4, which represents the finished proposals.
    # The most recent major update to the specification was approved in June 2015<sup>1</sup> and is called by many names: ECMAScript 6, ES6, ES2015, and ES6Harmony. Based on cur‐ rent plans, new specs will be released on a yearly cycle. The 2016 release was relatively small, but it already looks like ES2017 will include quite a few useful features. We'll be using many of these new features in the book and will opt to use emerging JavaScript whenever possible.
    # Many of these features are already supported by the newest browsers. We will also be covering how to convert your code from emerging JavaScript syntax to ES5 syntax that will work today in almost all browsers. The [kangax compatibility table](http://kangax.github.io/compat-table/esnext/) is a great place to stay informed about the latest JavaScript features and their varying degrees of support by browsers.
    # <sup>1</sup> Abel Avram, ["ECMAScript 2015 Has Been Approved",](http://bit.ly/2nvMJjJ) InfoQ, June 17, 2015.
    # In this chapter, we will show you all of the emerging JavaScript that we'll be using throughout the book. If you haven't made the switch to the latest syntax yet, now would be a good time to get started. If you are already comfortable with ES.Next lan‐ guage features, skip to the next chapter.
    # """
    # print(len(x.split("\n")))
