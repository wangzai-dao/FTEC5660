from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek

def build_chain() -> Any:
    # 1. 初始化 DeepSeek 视觉模型
    llm = ChatDeepSeek(model="deepseek-v4-flash-vision-exp", temperature=0)
    
    # 2. 创建多模态 Prompt 模板
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个精通分析超市小票的助手。请根据提供的所有小票图片，计算出用户的总金额。回答只需包含最终金额（如 HK$123.40），不要输出多余文字。"),
        ("human", [
            {"type": "text", "text": "{question}"},
            # 图片会通过 input 动态传入
        ])
    ])
    
    return prompt | llm

def answer_queries(chain: Any, images: list[Path]) -> dict[str, Any]:
    # 将图片转换为大模型需要的格式
    image_contents = [{"type": "image_url", "image_url": {"url": image_data_url(img)}} for img in images]
    
    responses = {}
    for query in QUERIES:
        # 将问题与所有图片合并送入 model/chain 求解
        input_data = {
            "question": query,
            # 根据你定义的 prompt 格式传入 image_contents
        }
        res = chain.invoke(...)
        responses[query] = res
        
    return responses
