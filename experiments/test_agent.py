import sys_config
from beta.agent import Agent
from beta.config import AgentConfig, AgentConfigChoseModel


def test_beta_agent():
    agent_config = AgentConfig(collection_name='test')
    AgentConfigChoseModel.chose_ollama_llm_model(agent_config, sys_config.OLLAMA_GRANITE_MODEL_4_3B_H)
    AgentConfigChoseModel.chose_ollama_embedding(
        config=agent_config,
        model_name=sys_config.OLLAMA_GRANITE_EMBEDDING_MODEL_768,
        dimensions=sys_config.OLLAMA_GRANITE_EMBEDDING_MODEL_DIMENSIONS
    )
    agent = Agent(agent_config)
    agent.use_ollama_llm()
    agent.use_ollama_embeddings()
    agent.generate_work_flow()

    q = 'Who you are? who made you?'
    q2 = 'What does LAMP stand for? '
    answer, docs_list, ref_list = agent.invoke_with_retrieved_contents(q2)
    print(answer)
    print('-----------------')
    for doc in docs_list:
        print('=======')
        print(doc)
    print('-----------------')
    for ref in ref_list:
        print(ref)


if __name__ == '__main__':
    test_beta_agent()
