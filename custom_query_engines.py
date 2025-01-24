from llama_index.core import PromptTemplate
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.response_synthesizers import BaseSynthesizer
from llama_index.llms.gemini import Gemini



class SummaryQueryEngine(CustomQueryEngine):
    retriever: BaseRetriever
    synthesizer: BaseSynthesizer
    llm: Gemini
    qa_prompt: PromptTemplate = PromptTemplate(
        '''

            "Context information is below.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"

            please go through the above case in complete detail and provide me a contextual summary of the case, A summary of the relevant legal information.\nI want large data. give large contextual summary full detailed case by going through the whole txt file. give proper detailed info of what all happened in the case arguments by appealant, responsdent, judge decisions, etc etc,\nI want everything in detail. all the facts in the case, all relevant sections in the file, all legal principles in the file, each and every cotext in the file \n \n 
                Follow this template for all the summaries like a bible:\n 
                1. Case Title

                •	Case Name: [e.g., ABC Corp. vs. XYZ Ltd.]
                •	Court: [e.g., Supreme Court of India]
                •	Date of Judgment: [e.g., 23rd August 2024]
                •	Citation: [e.g., 2024 SCC 123]

            2. Background and Context

                •	Brief Overview: A concise summary of the case background, including relevant events leading up to the legal dispute.
                •	Key Issues: A list of the main legal querys or issues presented in the case.

            3. Legal Principles Involved

                •	Relevant Statutes and Provisions: List of statutory provisions, rules, and regulations relevant to the case.
                •	Precedents Cited: Key past judgments cited during the case.
                •	Legal Doctrines: Any specific legal doctrines or principles applied in the judgment.

            4. Arguments Presented

                •	Plaintiff’s Argument: Summary of the arguments and claims made by the plaintiff.
                •	Defendant’s Argument: Summary of the arguments and defenses presented by the defendant.

            5. Court’s Analysis and Reasoning

                •	Key Findings: Important observations and findings made by the court.
                •	Interpretation of Law: How the court interpreted the relevant legal provisions and precedents.
                •	Application of Law: How the court applied the law to the facts of the case.

            6. Judgment

                •	Final Decision: The outcome of the case (e.g., in favor of the plaintiff/defendant).
                •	Relief Granted: Any relief or damages awarded, if applicable.
                •	Orders: Specific orders or directives issued by the court.

            7. Implications

                •	Impact on Law: How the judgment impacts existing law or legal practice.
                •	Future Relevance: The potential influence of the case on future legal decisions or cases.
                •	Broader Context: Any broader implications, such as social, economic, or political impact.

            8. Summary Points

                •	Key Takeaways: Bullet points summarizing the most critical aspects of the case, suitable for quick reference.

            9. References

                •	Citations: Full citations of any statutes, cases, or legal texts referenced in the summary.
                •	Further Reading: Suggestions for further reading or related cases.
            '''
        )

    def custom_query(self, query_str: str):
        nodes = self.retriever.retrieve(query_str)

        context_str = "\n\n".join([n.node.get_content() for n in nodes])
        response = self.llm.complete(
            self.qa_prompt.format(context_str=context_str, query_str=query_str)
    )

        return str(response)
    

class VectorQueryEngine(CustomQueryEngine):
    retriever: BaseRetriever
    synthesizer: BaseSynthesizer
    llm: Gemini
    qa_prompt: PromptTemplate = PromptTemplate(
        '''
        "Context information is below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"

        Go through the above context and answer the query below. If the query is not related to the context, just answer it based on your prior knowledge of the law, don't even mention that the query is not related to the context. Act as if the context is related to the query, but ignore the context while answering the query.
        
        You are a helpful, respectful, and honest legal research assistant. Answer the query using the context given to you.

        Your goal is to provide accurate legal research, relevant case law, statutory interpretation, and insights into legal principles and precedents, always maintaining a focus on legal accuracy and ethical standards.

        Answer the query based on the context below. If the query is not related to the context, just answer it based on your prior knowledge of the law, don't even mention that the query is not related to the context.

        Make sure to mention the previous relevant cases in the reasoning while crafting the response. If the query is not related to the context, still mention some relevant cases in the reasoning of your response on your own but don't mention that the context is not related to the query. Make sure if the cases you are mentioning in the response on your own and not present in the context then they should be Indian commercial cases only.

        Query:
        --------------------------------
        {query_str}
        --------------------------------
        '''
    )

    def custom_query(self, query_str: str):
        nodes = self.retriever.retrieve(query_str)

        context_str = "\n\n".join([n.node.get_content() for n in nodes])
        response = self.llm.complete(
            self.qa_prompt.format(context_str=context_str, query_str=query_str)
        )

        return str(response)