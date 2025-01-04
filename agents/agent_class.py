import os
from os.path import join
from collections import defaultdict
import pandas as pd
import re
import csv
from datetime import date
import argparse
from numpy.random import choice, shuffle
from agents.functions import *
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from agents.extract_belif import *
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain_google_genai import ChatGoogleGenerativeAI
import getpass
import os
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")


class Agent:
    def __init__(
        self, agent_id, persona, model_name, temperature, max_tokens, prompt_template_root
    ):
        """Constructor that initializes an object of type Agent

        Args:
            agent_id (int): Identification entity for the LLM Agent
            persona (str): Persona to be embodied by the LLM agent
        """
        # Initialize LLM agent, its identity, name, and persona by reading as step #1
        self.agent_id = agent_id
        self.prompt_template_root = prompt_template_root
        df_agents = pd.read_csv(join(self.prompt_template_root, "list_agent_descriptions.csv"))
        self.agent_name = str(df_agents.loc[self.agent_id - 1, "agent_name"])
        self.init_belief = df_agents.loc[self.agent_id - 1, "opinion"]
        self.current_belief = self.init_belief
        self.persona = persona
        self.count_tweet_written, self.count_tweet_seen = 0, 0
        self.previous_interaction_type = "none"

        persona_prompt = HumanMessagePromptTemplate.from_template(self.persona)
        if model_name == "gemini-pro":
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro",api_key=os.getenv("GOOGLE_API_KEY"))
        else:
            llm = ChatOpenAI(model_name=model_name, temperature=temperature, max_tokens=max_tokens, api_key=os.getenv("OPENAI_API_KEY"))

        # Create placeholders for back-and-forth history with the LLM
        history_message_holder = MessagesPlaceholder(variable_name="history")
        quesion_placeholder = HumanMessagePromptTemplate.from_template("{input}")

        with open(join(self.prompt_template_root, "step1_persona.md"), "r") as f:
            sys_prompt = f.read()
        sys_prompt = sys_prompt.split("\n---------------------------\n")[0].format(
            AGENT_PERSONA=self.persona, AGENT_NAME=self.agent_name
        )
        systems_prompt = SystemMessagePromptTemplate.from_template(sys_prompt)

        # Initialize the LLM agent with the language chain and its memory
        chat_prompt = ChatPromptTemplate.from_messages(
            [systems_prompt, persona_prompt, history_message_holder, quesion_placeholder]
        )
        memory = ConversationBufferMemory(
            return_messages=True, ai_prefix=self.agent_name, human_prefix="Game Master"
        )  # Add
        agent_conversation = ConversationChain(
            llm=llm, memory=memory, prompt=chat_prompt, verbose=True
        )

        self.memory = agent_conversation

    def receive_tweet(self, tweet, previous_interaction_type, tweet_written_count, add_to_memory):
        """Receive a tweet from another agent, and produce a response. The response contains the agent's updated opinion.

        Args:
            tweet (str): the tweet that the agent received.
            previous_interaction_type (str): the previous interaction type. The previous interaction type is either "tweet", "read", or "none" (no previous interaction so far).
            tweet_written_count (int): the number of tweets that the agent has written so far.
            add_to_memory (bool): whether to add the prompt and the produced tweet to the agent's memory. When using langchain's predict() function, it willby default add the prompt and the produced tweet to the agent's memory. When `add_to_memory=False`, the memory added by langchain will be removed.

        Returns:
            str: the response that the agent produced
        """
        assert previous_interaction_type in ["write", "read", "none"]

        if previous_interaction_type == "write":
            with open(
                join(
                    self.prompt_template_root,  "step3_receive_tweet_prev_tweet.md"
                ),
                "r",
            ) as f:
                prompt_instructions = f.read()

            prompt = prompt_instructions.split("\n---------------------------\n")[0].format(
                AGENT_NAME=self.agent_name,
                TWEET_WRITTEN_COUNT=tweet_written_count,
                SUPERSCRIPT=get_superscript(tweet_written_count),
                TWEET=tweet,
            )
            response = get_integer_llm_response(self.memory, prompt)

        elif previous_interaction_type == "read":
            with open(
                join(
                    self.prompt_template_root, "step3_receive_tweet_prev_read.md"
                ),
                "r",
            ) as f:
                prompt_instructions = f.read()

            prompt = prompt_instructions.split("\n---------------------------\n")[0].format(
                AGENT_NAME=self.agent_name,
                TWEET=tweet,
            )
            response = get_integer_llm_response(self.memory, prompt)

        elif previous_interaction_type == "none":
            with open(
                join(
                    self.prompt_template_root,  "step3_receive_tweet_prev_none.md"
                ),
                "r",
            ) as f:
                prompt_instructions = f.read()

            prompt = prompt_instructions.split("\n---------------------------\n")[0].format(
                AGENT_NAME=self.agent_name,
                TWEET=tweet,
            )
            response = get_integer_llm_response(self.memory, prompt)

        if not add_to_memory:
            self.memory.memory.chat_memory.messages.pop()
            self.memory.memory.chat_memory.messages.pop()

        return response

    def produce_tweet(self, previous_interaction_type, tweet_written_count, add_to_memory):
        """Produce a tweet based on the agent's opinion.

        Args:
            previous_interaction_type (str): the previous interaction type. The previous interaction type is either "tweet", "read", or "none" (no previous interaction so far).
            tweet_written_count (int): the number of tweets that the agent has written so far.
            add_to_memory (bool): whether to add the prompt and the produced tweet to the agent's memory. When using langchain's predict() function, it will by default add the prompt
                                and the produced tweet to the agent's memory. When `add_to_memory=False`, the memory added by langchain will be removed.
        Returns:
            str: the tweet that the agent produced
        """
        assert previous_interaction_type in ["write", "read", "none"]

        if previous_interaction_type == "write":
            with open(
                join(
                    self.prompt_template_root, "step2_produce_tweet_prev_tweet.md"
                ),
                "r",
            ) as f:
                prompt_instructions = f.read()

            prompt = prompt_instructions.split("\n---------------------------\n")[0].format(
                AGENT_NAME=self.agent_name,
                TWEET_WRITTEN_COUNT=tweet_written_count,
                SUPERSCRIPT=get_superscript(tweet_written_count),
            )
            tweet = get_llm_response(self.memory, prompt)

        elif previous_interaction_type == "read":
            with open(
                join(
                    self.prompt_template_root, "step2_produce_tweet_prev_read.md"
                ),
                "r",
            ) as f:
                prompt_instructions = f.read()

            prompt = prompt_instructions.split("\n---------------------------\n")[0].format(
                AGENT_NAME=self.agent_name,
            )
            tweet = get_llm_response(self.memory, prompt)

        elif previous_interaction_type == "none":
            with open(
                join(
                    self.prompt_template_root,  "step2_produce_tweet_prev_none.md"
                ),
                "r",
            ) as f:
                prompt_instructions = f.read()

            prompt = prompt_instructions.split("\n---------------------------\n")[0].format(
                AGENT_NAME=self.agent_name,
            )
            tweet = get_llm_response(self.memory, prompt)

        if not add_to_memory:
            self.memory.memory.chat_memory.messages.pop()
            self.memory.memory.chat_memory.messages.pop()

        return tweet

    def add_to_memory(
        self,
        previos_interaction_type,
        current_interaction_type,
        tweet_written_count,
        tweet_written=None,
        tweet_seen=None,
        response=None,rating_flag=False
    ):
        """Add the text to the agent's memory. The text can be either a tweet written or a tweet seen + a corresponding response. Note that this should not use the langchain's predict() function because we are only adding the text to the agent's memory rather than asking for a response.

        Args:
            previos_interaction_type (str): the previous interaction type. The interaction type is either "tweet","read", or "none".
            current_interaction_type (str): the current interaction type. The interaction type is either "tweet" or "read".
            tweet_written_count (int): the number of tweets that the agent has written so far.
            tweet_written (str): the tweet the agent saw. To be added to the agent's memory. Defaults to None. Required when `current_interaction_type="tweet"`.
            tweet_seen (str): the tweet the agent saw. To be added to the agent's memory. Defaults to None. Required when `current_interaction_type="read"`.
            response (str): the response that the agent produced. To be added to the agent's memory. Defaults to None. Required when `current_interaction_type="read"`.
        """
        if current_interaction_type == "write":
            assert tweet_written is not None
        elif current_interaction_type == "read":
            assert tweet_seen is not None and response is not None
        else:
            raise ValueError(
                f"current_interaction_type must be either 'write' or 'read'. Got {current_interaction_type}"
            )
        assert previos_interaction_type in ["write", "read", "none"]
        assert current_interaction_type in ["write", "read"]

        if previos_interaction_type == "write":
            if current_interaction_type == "write":
                with open(
                    join(
                        self.prompt_template_root,
                        "step2b_add_to_memory_prev_tweet_cur_tweet.md",
                    ),
                    "r",
                ) as f:
                    prompt_instructions = f.read()

                prompt = prompt_instructions.split("\n---------------------------\n")[0].format(
                    TWEET_WRITTEN_COUNT_LAST=tweet_written_count - 1,
                    SUPERSCRIPT_LAST=get_superscript(tweet_written_count - 1),
                    TWEET_WRITTEN_COUNT=tweet_written_count,
                    SUPERSCRIPT=get_superscript(tweet_written_count),
                    TWEET_WRITTEN=tweet_written,
                )
                self.memory.memory.chat_memory.add_user_message(prompt)

            elif current_interaction_type == "read":
                with open(
                    join(
                        self.prompt_template_root,
                        "step2b_add_to_memory_prev_tweet_cur_read.md",
                    ),
                    "r",
                ) as f:
                    prompt_instructions = f.read()

                if not rating_flag:
                    belief = extract_belief(response)
                    reasoning = extract_reasoning(response)
                    prompt = prompt_instructions.split("\n---------------------------\n")[0].format(
                        TWEET_WRITTEN_COUNT=tweet_written_count,
                        SUPERSCRIPT=get_superscript(tweet_written_count),
                        TWEET_SEEN=tweet_seen,
                        REASONING=reasoning,
                        BELIEF_RATING=belief,
                    )
                else:
                    reasoning = extract_reasoning(response)
                    prompt = prompt_instructions.split("\n---------------------------\n")[0].format(
                        TWEET_WRITTEN_COUNT=tweet_written_count,
                        SUPERSCRIPT=get_superscript(tweet_written_count),
                        TWEET_SEEN=tweet_seen,
                        REASONING=reasoning,
                    )
                self.memory.memory.chat_memory.add_user_message(prompt)

        elif previos_interaction_type == "read":
            if current_interaction_type == "write":
                with open(
                    join(
                        self.prompt_template_root,
                        "step2b_add_to_memory_prev_read_cur_tweet.md",
                    ),
                    "r",
                ) as f:
                    prompt_instructions = f.read()

                prompt = prompt_instructions.split("\n---------------------------\n")[0].format(
                    TWEET_WRITTEN_COUNT=tweet_written_count,
                    SUPERSCRIPT=get_superscript(tweet_written_count),
                    TWEET_WRITTEN=tweet_written,
                )
                self.memory.memory.chat_memory.add_user_message(prompt)

            elif current_interaction_type == "read":
                with open(
                    join(
                        self.prompt_template_root,
                        "step2b_add_to_memory_prev_read_cur_read.md",
                    ),
                    "r",
                ) as f:
                    prompt_instructions = f.read()

                if not rating_flag:
                    belief = extract_belief(response)
                    reasoning = extract_reasoning(response)
                    prompt = prompt_instructions.split("\n---------------------------\n")[0].format(
                        TWEET_SEEN=tweet_seen, REASONING=reasoning, BELIEF_RATING=belief
                    )
                else:
                    reasoning = extract_reasoning(response)
                    prompt = prompt_instructions.split("\n---------------------------\n")[0].format(
                        TWEET_SEEN=tweet_seen, REASONING=reasoning
                    )
                self.memory.memory.chat_memory.add_user_message(prompt)

        elif previos_interaction_type == "none":
            if current_interaction_type == "write":
                with open(
                    join(
                        self.prompt_template_root,
                        "step2b_add_to_memory_prev_none_cur_tweet.md",
                    ),
                    "r",
                ) as f:
                    prompt_instructions = f.read()

                prompt = prompt_instructions.split("\n---------------------------\n")[0].format(
                    TWEET_WRITTEN_COUNT=tweet_written_count,
                    SUPERSCRIPT=get_superscript(tweet_written_count),
                    TWEET_WRITTEN=tweet_written,
                )
                self.memory.memory.chat_memory.add_user_message(prompt)

            elif current_interaction_type == "read":
                with open(
                    join(
                        self.prompt_template_root,
                        "step2b_add_to_memory_prev_none_cur_read.md",
                    ),
                    "r",
                ) as f:
                    prompt_instructions = f.read()

                if not rating_flag:
                    belief = extract_belief(response)
                    reasoning = extract_reasoning(response)
                    prompt = prompt_instructions.split("\n---------------------------\n")[0].format(
                        TWEET_SEEN=tweet_seen, REASONING=reasoning, BELIEF_RATING=belief
                    )
                else:
                    reasoning = extract_reasoning(response)
                    prompt = prompt_instructions.split("\n---------------------------\n")[0].format(
                        TWEET_SEEN=tweet_seen, REASONING=reasoning
                    )
                self.memory.memory.chat_memory.add_user_message(prompt)

    def get_count_tweet_written(self):
        """Get the number of tweets that the agent has written so far.

        Returns:
            int: the number of tweets that the agent has written so far.
        """
        return self.count_tweet_written

    def increase_count_tweet_written(self):
        """Increase the number of tweets that the agent has written so far by 1."""
        self.count_tweet_written += 1

    def get_count_tweet_seen(self):
        """Get the number of tweets that the agent has seen so far.

        Returns:
            int: the number of tweets that the agent has seen so far.
        """
        return self.count_tweet_seen

    def increase_count_tweet_seen(self):
        """Increase the number of tweets that the agent has seen so far by 1."""
        self.count_tweet_seen += 1

    def outdate_persona_memory(self):
        """Outdate the persona memory. E.g., use past tense to describe the persona memory. Should "rewrite" the agent's memory."""
        self.persona = convert_text_from_present_to_past(self.persona)