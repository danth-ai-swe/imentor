class Reflection:
    def __init__(self, llm):
        self.llm = llm

    @staticmethod
    def _concat_and_format_texts(data):
        concatenated_texts = []
        all_texts= ''
        for entry in data:
            role = entry.get('role', '')
            if entry.get('parts'):
                all_texts = ' '.join(part['text'] for part in entry['parts'] )
            elif entry.get('content'):
                all_texts = entry['content'] 
            concatenated_texts.append(f"{role}: {all_texts} \n")
        return ''.join(concatenated_texts)


    def __call__(self, chat_history, last_items_considered=100):
        
        if len(chat_history) >= last_items_considered:
            chat_history = chat_history[len(chat_history) - last_items_considered:]

        history_string = self._concat_and_format_texts(chat_history)

        higher_level_summaries_prompt = {
            "role": "user",
            "content": """Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question in Vietnamese which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is. {historyString}
        """.format(historyString=history_string)
        }

        print(higher_level_summaries_prompt)

        completion = self.llm.generate_content([higher_level_summaries_prompt])
    
        return completion

