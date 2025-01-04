import re
from random import choice, shuffle
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

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

def get_llm_response(conversation, prompt):
    return conversation.predict(input=prompt)


def get_integer_llm_response(conversation, prompt):
    max_attempts = 3
    attempts = 0

    while attempts < max_attempts:
        response = get_llm_response(conversation, prompt)
        try:
            if len(re.findall("[+-]?([0-9]*)?[.][0-9]+", response)) > 0:
                print("Bad (non-integral) value found, re-prompting agent...")
                raise ValueError
            break
        except ValueError:
            attempts += 1
            if attempts == 3:
                import pdb

                pdb.set_trace()
            conversation.memory.chat_memory.messages.pop()

    if attempts == max_attempts:
        print("Failed to get a valid integer Likert scale belief after 3 attempts.")
        raise ValueError

    return response


def get_random_pair(list_agents):
    """Get a random pair of agents from the list of agents.

    Args:
        list_agents (list(Agent)): A list of agents

    Returns:
        tuple: A tuple of agents (agent_i, agent_j)
    """
    size = len(list_agents)
    index_list = list(range(size))

    agent_idx_i = choice(index_list)
    index_list.remove(agent_idx_i)
    agent_idx_j = choice(index_list)

    agent_i, agent_j = list_agents[agent_idx_i], list_agents[agent_idx_j]
    return agent_i, agent_j


def initialize_opinion_distribution(num_agents, list_opinion_space, distribution_type="uniform"):
    """Initialize the opinion distribution of the agents.

    Args:
        num_agents (int): number of agents
        list_opinion_space (list(int), optional): the range of the opinion space. For example, [-3,-2, ..., 2, 3].
        distribution_type (str, optional): the type of distribution. Defaults to "uniform".

    Returns:
        list: a list of opinions for each agent
    """
    if distribution_type == "uniform":
        multiple = num_agents // 5
        list_opinions = list_opinion_space * multiple
        shuffle(list_opinions)
    else:
        raise NotImplementedError
    return list_opinions



def get_superscript(count):
    if count in [1, 21]:
        return "st"
    elif count in [2, 22]:
        return "nd"
    elif count in [3, 23]:
        return "rd"
    elif count in [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]:
        return "th"


def initialize_opinion_distribution(num_agents, list_opinion_space, distribution_type="uniform"):
    """Initialize the opinion distribution of the agents.

    Args:
        num_agents (int): number of agents
        list_opinion_space (list(int), optional): the range of the opinion space. For example, [-3,-2, ..., 2, 3].
        distribution_type (str, optional): the type of distribution. Defaults to "uniform".

    Returns:
        list: a list of opinions for each agent
    """
    max_opinion = max(list_opinion_space)
    min_opinion = min(list_opinion_space)
    multiple = num_agents // 5
    if distribution_type == "uniform":
        list_opinions = list_opinion_space * multiple
    elif distribution_type == "skewed_positive":
        list_opinions = [max_opinion] * (num_agents - multiple) + [min_opinion] * multiple
    elif distribution_type == "skewed_negative":
        list_opinions = [min_opinion] * (num_agents - multiple) + [max_opinion] * multiple
    elif distribution_type == "positive":
        list_opinions = [max_opinion] * num_agents
    elif distribution_type == "negative":
        list_opinions = [min_opinion] * num_agents
    else:
        raise NotImplementedError
    shuffle(list_opinions)
    return list_opinions

import spacy
import pyinflect
def convert_text_from_present_to_past(text):
    """Convert the text from present tense to past tense.

    Args:
        text (str): the text in present

    Returns:
        str: the text in past
    """

    nlp = spacy.load("en_core_web_sm")
    present_tense_doc = nlp(text)

    for i in range(len(present_tense_doc)):
        token = present_tense_doc[i]
        if token.tag_ in ["VBP", "VBZ"]:
            text = text.replace(token.text, token._.inflect("VBD"))

    converted_text = text
    return converted_text


def get_random_pair(list_agents):
    """Get a random pair of agents from the list of agents.

    Args:
        list_agents (list(Agent)): A list of agents

    Returns:
        tuple: A tuple of agents (agent_i, agent_j)
    """
    size = len(list_agents)
    index_list = list(range(size))

    agent_idx_i = choice(index_list)
    index_list.remove(agent_idx_i)
    agent_idx_j = choice(index_list)

    agent_i, agent_j = list_agents[agent_idx_i], list_agents[agent_idx_j]
    return agent_i, agent_j


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(50))
def get_llm_response(conversation, prompt):
    return conversation.predict(input=prompt)


def get_integer_llm_response(conversation, prompt):
    max_attempts = 3
    attempts = 0

    while attempts < max_attempts:
        response = get_llm_response(conversation, prompt)
        try:
            if len(re.findall("[+-]?([0-9]*)?[.][0-9]+", response)) > 0:
                print("Bad (non-integral) value found, re-prompting agent...")
                raise ValueError
            break
        except ValueError:
            attempts += 1
            if attempts == 3:
                import pdb

                pdb.set_trace()
            conversation.memory.chat_memory.messages.pop()

    if attempts == max_attempts:
        print("Failed to get a valid integer Likert scale belief after 3 attempts.")
        raise ValueError

    return response


def main(
    num_agents,
    num_steps,
    experiment_id,
    model_name,
    temperature,
    max_tokens,
    opinion_space,
    prompt_template_root,
    date_version,
    path_result,
):
    prompt_template_root = os.path.join(prompt_template_root, experiment_id)
    list_opinion = initialize_opinion_distribution(
        num_agents=num_agents, list_opinion_space=opinion_space, distribution_type=args.distribution
    )

    # Reading in the list of agents and creating a dataframe of selected agents
    if args.distribution == "uniform":
        df_agents = pd.read_csv(join(prompt_template_root, "list_agent_descriptions_neutral.csv"))
    else:
        df_agents = pd.read_csv(
            join(prompt_template_root, "list_agent_descriptions_neutral_special.csv")
        )

    df_agents_selected = pd.DataFrame()

    for opinion in list_opinion:
        # for each opinion, select one agent with that opinion (without replacement)
        df_agent_candidate = df_agents[df_agents["opinion"] == opinion].sample(n=1)
        if not df_agents_selected.empty:
            # If the agent was selected before, resample from the agent dataframe
            while df_agent_candidate["agent_id"].values[0] in df_agents_selected["agent_id"].values:
                df_agent_candidate = df_agents[df_agents["opinion"] == opinion].sample(
                    n=1, replace=False
                )

        # Add the selected agent to the dataframe
        df_agents_selected = pd.concat([df_agents_selected, df_agent_candidate])

    # create a list of agent objects
    list_agents = []
    list_agent_ids = df_agents_selected["agent_id"]
    list_agent_persona = df_agents_selected["persona"]
    for persona, agent_id in zip(list_agent_persona, list_agent_ids):
        agent = Agent(agent_id, persona, model_name, temperature, max_tokens, prompt_template_root)
        # And add them to the list of agents
        list_agents.append(agent)

    # ------------------
    # results collectors
    # ------------------
    # - dict_agent_report {agent_id: [(t1,report_1),(t6,report_6),(t12,report_12)]}
    dict_agent_report = defaultdict(list)

    for t in range(num_steps):
        # ------------------
        # iterate over each agent
        # ------------------
        _, agent_j = get_random_pair(list_agents)

        if agent_j.get_count_opinion_reported() == 1:
            agent_j.outdate_persona_memory()
        # ------------------
        # agent j reports its opinion
        # - ask for j's opinion
        # - the prompt is different depending on agent j's interaction history: 1) no history, 2) previous interaction = report opinion
        # ------------------
        agent_j_previos_interaction_type = agent_j.previous_interaction_type
        if agent_j_previos_interaction_type in ["none", "report"]:
            report_j = agent_j.report_opinion(
                previous_interaction_type=agent_j_previos_interaction_type,
                report_count=agent_j.get_count_opinion_reported(),
                add_to_memory=False,
            )
            agent_j.increase_count_opinion_reported()
        else:
            raise ValueError(
                "agent_j_previos_interaction_type is not valid: {}".format(
                    agent_j_previos_interaction_type
                )
            )

        agent_j.add_to_memory(
            opinion_reported=report_j,
            previos_interaction_type=agent_j_previos_interaction_type,
            current_interaction_type="report",
            report_count=agent_j.get_count_opinion_reported(),
        )

        agent_j.previous_interaction_type = "report"

        # ------------------
        # end of the interaction, save to result collectors
        # ------------------
        dict_agent_report[agent_j].append((t + 1, report_j))

    return (
        post_process_memory(list_agents, path_result, date_version),
        post_process_report(dict_agent_report, path_result, date_version),
    )


def post_process_memory(list_agents, path_result, date_version):
    for agent in list_agents:
        out_name = os.path.join(
            path_result,
            "log_conversation",
            args.output_file.split(".cs")[0]
            + "_"
            + str(args.num_agents)
            + "_"
            + str(args.num_steps)
            + "_"
            + args.version_set
            + "_"
            + date_version
            + "_agent_"
            + str(agent.agent_id)
            + "_"
            + args.distribution
            + "_reflection.txt",
        )
        with open(out_name, "w+") as f:
            f.write(agent.memory.prompt.messages[0].prompt.template)
            f.write("\n------------------------------\n")
            for i in range(len(agent.memory.memory.chat_memory.messages)):
                f.write(agent.memory.memory.chat_memory.messages[i].content)
                f.write("\n------------------------------\n")

    return


def post_process_report(dict_agent_report, path_result, date_version):
    if not os.path.exists(path_result):
        os.makedirs(path_result)
        print("Created a fresh directory!")

    # Create a new directory because it does not exist
    out_name = (
        os.path.join(path_result, args.output_file.split(".cs")[0])
        + "_"
        + str(args.num_agents)
        + "_"
        + str(args.num_steps)
        + "_"
        + args.version_set
        + "_"
        + date_version
        + "_agent_tweet_history_"
        + args.distribution
        + "_reflection.csv"
    )
    with open(out_name, "w+") as g:
        writer = csv.writer(g, delimiter=",")
        writer.writerow(
            [
                "Agent Name",
                "Original Belief",
                "Belief Changes Time Step",
                "Response Chain",
            ]
        )

        for agent in dict_agent_report.keys():
            time_step_changes = [time_step[0] for time_step in dict_agent_report[agent]]
            opinion_chain = [time_step[1] for time_step in dict_agent_report[agent]]
            row = []
            row.append(agent.agent_name)
            row.append(agent.init_belief)
            row.append(list(time_step_changes))
            row.append(list(opinion_chain))

            writer.writerow(row)

    return


def post_process_memory(list_agents, path_result, date_version):
    for agent in list_agents:
        out_name = os.path.join(
            path_result,
            "log_conversation",
            args.output_file.split(".cs")[0]
            + "_"
            + str(args.num_agents)
            + "_"
            + str(args.num_steps)
            + "_"
            + args.version_set
            + "_"
            + date_version
            + "_agent_"
            + str(agent.agent_id)
            + "_"
            + args.distribution
            + ".txt",
        )
        with open(out_name, "w+") as f:
            f.write(agent.memory.prompt.messages[0].prompt.template)
            f.write("\n------------------------------\n")
            for i in range(len(agent.memory.memory.chat_memory.messages)):
                f.write(agent.memory.memory.chat_memory.messages[i].content)
                f.write("\n------------------------------\n")

    return


def post_process_tweet(dict_agent_tweet, path_result, date_version):
    if not os.path.exists(path_result):
        os.makedirs(path_result)
        print("Created a fresh directory!")

    # Create a new directory because it does not exist
    out_name = (
        os.path.join(path_result, args.output_file.split(".cs")[0])
        + "_"
        + str(args.num_agents)
        + "_"
        + str(args.num_steps)
        + "_"
        + args.version_set
        + "_"
        + date_version
        + "_agent_tweet_history_"
        + args.distribution
        + ".csv"
    )
    with open(out_name, "w+") as g:
        writer = csv.writer(g, delimiter=",")
        writer.writerow(
            [
                "Agent Name",
                "Original Belief",
                "Tweet Time Step",
                "Belief When Tweeting",
                "Tweet Chain",
            ]
        )

        for agent in dict_agent_tweet.keys():
            row = []
            row.append(agent.agent_name)
            row.append(agent.init_belief)

            time_step_changes = [time_step[0] for time_step in dict_agent_tweet[agent]]
            belief_when_tweeting = [time_step[1] for time_step in dict_agent_tweet[agent]]
            tweet_chain = [time_step[2] for time_step in dict_agent_tweet[agent]]

            row.append(list(time_step_changes))
            row.append(list(belief_when_tweeting))
            row.append(list(tweet_chain))

            writer.writerow(row)

    return


def post_process_response(dict_agent_response, path_result, date_version):
    if not os.path.exists(path_result):
        os.makedirs(path_result)
        print("Created a fresh directory!")

    # Create a new directory because it does not exist
    out_name = (
        os.path.join(path_result, args.output_file.split(".cs")[0])
        + "_"
        + str(args.num_agents)
        + "_"
        + str(args.num_steps)
        + "_"
        + args.version_set
        + "_"
        + date_version
        + "_agent_response_history_"
        + args.distribution
        + ".csv"
    )
    with open(out_name, "w+") as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerow(
            [
                "Agent Name",
                "Original Belief",
                "Belief Changes Time Step",
                "Belief Change Chain",
                "Response chain",
            ]
        )

        for agent in dict_agent_response.keys():
            row = []
            row.append(agent.agent_name)
            row.append(agent.init_belief)

            time_step_changes = [time_step[0] for time_step in dict_agent_response[agent]]
            belief_changes = [
                extract_belief(time_step[1]) for time_step in dict_agent_response[agent]
            ]
            response_chain = [time_step[1] for time_step in dict_agent_response[agent]]

            row.append(list(time_step_changes))
            row.append(list(belief_changes))
            row.append(list(response_chain))

            writer.writerow(row)
    return






def extract_reasoning(tweet, rating_flag=False):
    if not rating_flag:
        reasoning = tweet.split("\nFinal Answer")[0]
    else:
        reasoning = "Reasoning:" + tweet.split("\nReasoning")[-1]
    return reasoning