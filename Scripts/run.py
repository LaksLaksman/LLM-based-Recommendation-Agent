import time

def Recommend_movies(message: str) -> str:
    start_time=time.time()

    from pinecone import Pinecone
    import os
    from llama_index.vector_stores.pinecone import PineconeVectorStore
    from llama_index.core import VectorStoreIndex,Settings
    from llama_index.embeddings.ollama import OllamaEmbedding
    from llama_index.llms.ollama import Ollama

    ### setting embedd model
    ollama_embedding = OllamaEmbedding(
    # model_name="llama3.2:latest",
    model_name="mxbai-embed-large:latest",
    base_url="http://localhost:11434",
    ollama_additional_kwargs={"mirostat": 0},
    )

    #set llm
    llm = Ollama(model="llama3.1:latest", request_timeout=120.0)

    Settings.llm = llm
    Settings.embed_model = ollama_embedding


    #setting pinecone

    os.environ[   
    "PINECONE_API_KEY"
    ] = "your api key"

    api_key = os.environ["PINECONE_API_KEY"]

    pc = Pinecone(api_key=api_key)
    index_name = "moviedata" # write your index name

    # Initialize your index
    pinecone_index = pc.Index(index_name)

    # Initialize VectorStore
    vector_store = PineconeVectorStore(batch_size=50 ,pinecone_index=pinecone_index) #batch_size=50 ,

    #set index
    index = VectorStoreIndex.from_vector_store(vector_store)

    #query
    query_engine = index.as_query_engine()

    Retriever_prompt_tmpl_str=(
        "You are an expert recommendation system that can pick the top 5 best recommended movies.\n"
        "a list of movies watched by a person recently are given in the query_str.\n"
        "follow the below steps to give the best recommendations \n"
        "1. grab all the possible details about the movies listed in the query_str.\n "
        "2. understand the taste of the person by making the best summary about those movies watched, such as what genere he likes most, what kind of story he enjoys, finding commonness in all the movies may be the director, may be he is interested in watching high rated movies.\n "
        "3. now find the best 5 recommended movies to suggest to the person.\n"
        "4. never recommend any movie that are beyond provided context.\n"
        "5. justify your recommendations why you suggest those movies"

        "Context information is below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Given the context information and not prior knowledge, "
        "find the best 5 recommendation movies.\n"
        "Query: {query_str}\n"
        "Answer: "

    )

    from llama_index.core import PromptTemplate
    # qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)
    Retriever_prompt_tmpl=PromptTemplate(Retriever_prompt_tmpl_str)

    query_engine.update_prompts(
    {"response_synthesizer:text_qa_template": Retriever_prompt_tmpl}
    )

    response = query_engine.query(message)
    # print(response)

    end_time = time.time()
    execution_time = end_time - start_time  # Calculate execution time

    return f"Processed: {response}"


if __name__ == "__main__":

    test_movielist = "Grandma and machine gun,Da Neve A Rosa,I Want To Do It With Sola Aoi!,Nirvana Live at the Paradiso,Bilge Ana Mevlüde Genç"
    print(Recommend_movies(test_movielist))
