"""
A multi-agent conversational system for navigating a complex task tree.
In this demo, the user is navigating a financial system.

Agent 1: look up a stock price.
Does not require authentication, has specialized tools for looking up stock prices.

Agent 2: authenticate the user.
A required step before the user can interact with some of the other agents.

Agent 3: look up an account balance.
Requires the user to be authenticated first.

Agent 4: transfer money between accounts.
Requires the user to be authenticated first, and have looked up their balance already.

Concierge agent: a catch-all agent that helps navigate between the other 4.

Orchestration agent: decides which agent to run based on the current state of the user.
"""

from dotenv import load_dotenv
load_dotenv()

from enum import Enum
from typing import List
import pprint
from colorama import Fore, Back, Style

from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI 
from llama_index.agent.openai import OpenAIAgent


class Speaker(str, Enum):
    STOCK_LOOKUP = "stock_lookup"
    AUTHENTICATE = "authenticate"
    ACCOUNT_BALANCE = "account_balance"
    TRANSFER_MONEY = "transfer_money"
    CONCIERGE = "concierge"
    ORCHESTRATOR = "orchestrator"

# Stock lookup agent
def stock_lookup_agent_factory(state: dict) -> OpenAIAgent:
    
    def lookup_stock_price(stock_symbol: str) -> str:
        """Useful for looking up a stock price."""
        print(f"Looking up stock price for {stock_symbol}")
        return f"Symbol {stock_symbol} is currently trading at $100.00"
    
    def search_for_stock_symbol(str: str) -> str:
        """Useful for searching for a stock symbol given a free-form company name."""
        print("Searching for stock symbol")
        return str.upper()
    
    def done() -> None:
        """When you have returned a stock price, call this tool."""
        print("Stock lookup is complete")
        state["current_speaker"] = None
        state["just_finished"] = True
    
    tools = [
        FunctionTool.from_defaults(fn=lookup_stock_price),
        FunctionTool.from_defaults(fn=search_for_stock_symbol),
        FunctionTool.from_defaults(fn=done),
    ]

    system_prompt = (f"""
        You are a helpful assistant that is looking up stock prices.
        The user may not know the stock symbol of the company they're interested in,
        so you can help them look it up by the name of the company.
        You can only look up stock symbols given to you by the search_for_stock_symbol tool, don't make them up. Trust the output of the search_for_stock_symbol tool even if it doesn't make sense to you.
        The current user state is:
        {pprint.pformat(state, indent=4)}
        Once you have supplied a stock price, you must call the tool "done" to signal that you are done.
        If the user asks to do anything other than look up a stock symbol or price, call the tool "done" to signal some other agent should help.
    """)

    return OpenAIAgent.from_tools(
        tools,
        llm=OpenAI(model="gpt-4o"),
        system_prompt=system_prompt,
    )

# Auth Agent
def auth_agent_factory(state: dict) -> OpenAIAgent:

    def store_username(username: str) -> None:
        """Adds the username to the user state."""
        print("Recording username")
        state["username"] = username

    def login(password: str) -> None:
        """Given a password, logs in and stores a session token in the user state."""
        print(f"Logging in {state['username']}")
        # todo: actually check the password
        session_token = "output_of_login_function_goes_here"
        state["session_token"] = session_token
    
    def is_authenticated() -> bool:
        """Checks if the user has a session token."""
        print("Checking if authenticated")
        if state["session_token"] is not None:
            return True

    def done() -> None:
        """When you complete your task, call this tool."""
        print("Authentication is complete")
        state["current_speaker"] = None
        state["just_finished"] = True

    tools = [
        FunctionTool.from_defaults(fn=store_username),
        FunctionTool.from_defaults(fn=login),
        FunctionTool.from_defaults(fn=is_authenticated),
        FunctionTool.from_defaults(fn=done),
    ]

    system_prompt = (f"""
        You are a helpful assistant that is authenticating a user.
        Your task is to get a valid session token stored in the user state.
        To do this, the user must supply you with a username and a valid password. You can ask them to supply these.
        If the user supplies a username and password, call the tool "login" to log them in.
        The current user state is:
        {pprint.pformat(state, indent=4)}
        When you have authenticated, call the tool "done" to signal that you are done.
        If the user asks to do anything other than authenticate, call the tool "done" to signal some other agent should help.
    """)

    return OpenAIAgent.from_tools(
        tools,
        llm=OpenAI(model="gpt-4o"),
        system_prompt=system_prompt,
    )

# Account balance agent
def account_balance_agent_factory(state: dict) -> OpenAIAgent:

    def get_account_id(account_name: str) -> str:
        """Useful for looking up an account ID."""
        print(f"Looking up account ID for {account_name}")
        account_id = "1234567890"
        state["account_id"] = account_id
        return f"Account id is {account_id}"
    
    def get_account_balance(account_id: str) -> str:
        """Useful for looking up an account balance."""
        print(f"Looking up account balance for {account_id}")
        state["account_balance"] = 1000
        return f"Account {account_id} has a balance of ${state['account_balance']}"
    
    def is_authenticated() -> bool:
        """Checks if the user has a session token."""
        print("Account balance agent is checking if authenticated")
        if state["session_token"] is not None:
            return True

    def done() -> None:
        """When you complete your task, call this tool."""
        print("Account balance lookup is complete")
        state["current_speaker"] = None
        state["just_finished"] = True
    
    tools = [
        FunctionTool.from_defaults(fn=get_account_id),
        FunctionTool.from_defaults(fn=get_account_balance),
        FunctionTool.from_defaults(fn=is_authenticated),
        FunctionTool.from_defaults(fn=done),
    ]

    system_prompt = (f"""
        You are a helpful assistant that is looking up account balances.
        The user may not know the account ID of the account they're interested in,
        so you can help them look it up by the name of the account.
        The user can only do this if they are authenticated, which you can check with the is_authenticated tool.
        If they aren't authenticated, tell them to authenticate
        If they're trying to transfer money, they have to check their account balance first, which you can help with.
        The current user state is:
        {pprint.pformat(state, indent=4)}
        Once you have supplied an account balance, you must call the tool "done" to signal that you are done.
        If the user asks to do anything other than look up an account balance, call the tool "done" to signal some other agent should help.
    """)

    return OpenAIAgent.from_tools(
        tools,
        llm=OpenAI(model="gpt-4o"),
        system_prompt=system_prompt,
    )

def transfer_money_agent_factory(state: dict) -> OpenAIAgent:
    
    def transfer_money(from_account_id: str, to_account_id: str, amount: int) -> None:
        """Useful for transferring money between accounts."""
        print(f"Transferring {amount} from {from_account_id} account {to_account_id}")
        return f"Transferred {amount} to account {to_account_id}"
    
    def balance_sufficient(account_id: str, amount: int) -> bool:
        """Useful for checking if an account has enough money to transfer."""
        # todo: actually check they've selected the right account ID
        print("Checking if balance is sufficient")
        if state['account_balance'] >= amount:
            return True
        
    def has_balance() -> bool:
        """Useful for checking if an account has a balance."""
        print("Checking if account has a balance")
        if state["account_balance"] is not None:
            return True
    
    def is_authenticated() -> bool:
        """Checks if the user has a session token."""
        print("Transfer money agent is checking if authenticated")
        if state["session_token"] is not None:
            return True

    def done() -> None:
        """When you complete your task, call this tool."""
        print("Money transfer is complete")
        state["current_speaker"] = None
        state["just_finished"] = True
    
    tools = [
        FunctionTool.from_defaults(fn=transfer_money),
        FunctionTool.from_defaults(fn=balance_sufficient),
        FunctionTool.from_defaults(fn=has_balance),
        FunctionTool.from_defaults(fn=is_authenticated),
        FunctionTool.from_defaults(fn=done),
    ]

    system_prompt = (f"""
        You are a helpful assistant that transfers money between accounts.
        The user can only do this if they are authenticated, which you can check with the is_authenticated tool.
        If they aren't authenticated, tell them to authenticate first.
        The user must also have looked up their account balance already, which you can check with the has_balance tool.
        If they haven't already, tell them to look up their account balance first.
        The current user state is:
        {pprint.pformat(state, indent=4)}
        Once you have transferred the money, you can call the tool "done" to signal that you are done.
        If the user asks to do anything other than transfer money, call the tool "done" to signal some other agent should help.
    """)

    return OpenAIAgent.from_tools(
        tools,
        llm=OpenAI(model="gpt-4o"),
        system_prompt=system_prompt,
    )

# Concierge agent
def concierge_agent_factory(state: dict) -> OpenAIAgent:

    def dummy_tool() -> bool:
        """A tool that does nothing."""
        print("Doing nothing.")

    tools = [
        FunctionTool.from_defaults(fn=dummy_tool)
    ]

    system_prompt = (f"""
        You are a helpful assistant that is helping a user navigate a financial system.
        Your job is to ask the user questions to figure out what they want to do, and give them the available things they can do.
        That includes
        * looking up a stock price            
        * authenticating the user
        * checking an account balance (requires authentication first)
        * transferring money between accounts (requires authentication and checking an account balance first)

        The current state of the user is:
        {pprint.pformat(state, indent=4)}
    """)

    return OpenAIAgent.from_tools(
        tools,
        llm=OpenAI(model="gpt-4o"),
        system_prompt=system_prompt,
    )

# Continuation agent
def continuation_agent_factory(state: dict) -> OpenAIAgent:
    
    def dummy_tool() -> bool:
        """A tool that does nothing."""
        print("Doing nothing.")

    tools = [
        FunctionTool.from_defaults(fn=dummy_tool)
    ]

    system_prompt = (f"""
        The current state of the user is:
        {pprint.pformat(state, indent=4)}
    """)

    return OpenAIAgent.from_tools(
        tools,
        llm=OpenAI(model="gpt-4o",temperature=0.4),
        system_prompt=system_prompt,
    )

# Orchestration agent
def orchestration_agent_factory(state: dict) -> OpenAIAgent:

    def has_balance() -> bool:
        """Useful for checking if an account has a balance."""
        print("Orchestrator checking if account has a balance")
        return (state["account_balance"] is not None)
    
    def is_authenticated() -> bool:
        """Checks if the user has a session token."""
        print("Orchestrator is checking if authenticated")
        return (state["session_token"] is not None)

    tools = [
        FunctionTool.from_defaults(fn=has_balance),
        FunctionTool.from_defaults(fn=is_authenticated),
    ]
    
    system_prompt = (f"""
        You are on orchestration agent.
        Your job is to decide which agent to run based on the current state of the user and what they've asked to do. Agents are identified by short strings.
        What you do is return the name of the agent to run next. You do not do anything else.
        
        The current state of the user is:
        {pprint.pformat(state, indent=4)}

        If a current_speaker is already selected in the state, simply output that value.

        If there is no current_speaker value, look at the chat history and the current state and you MUST return one of these strings identifying an agent to run:
        * "{Speaker.STOCK_LOOKUP.value}" - if they user wants to look up a stock price (does not require authentication)
        * "{Speaker.AUTHENTICATE.value}" - if the user needs to authenticate
        * "{Speaker.ACCOUNT_BALANCE.value}" - if the user wants to look up an account balance
            * If they want to look up an account balance, but they haven't authenticated yet, return "{Speaker.AUTHENTICATE.value}" instead
        * "{Speaker.TRANSFER_MONEY.value}" - if the user wants to transfer money between accounts (requires authentication and checking an account balance first)
            * If they want to transfer money, but is_authenticated returns false, return "{Speaker.AUTHENTICATE.value}" instead
            * If they want to transfer money, but has_balance returns false, return "{Speaker.ACCOUNT_BALANCE.value}" instead
        * "{Speaker.CONCIERGE.value}" - if the user wants to do something else, or hasn't said what they want to do, or you can't figure out what they want to do. Choose this by default.

        Output one of these strings and ONLY these strings, without quotes.
        NEVER respond with anything other than one of the above five strings. DO NOT be helpful or conversational.
    """)

    return OpenAIAgent.from_tools(
        tools,
        llm=OpenAI(model="gpt-4o",temperature=0.4),
        system_prompt=system_prompt,
    )

def get_initial_state() -> dict:
    return {
        "username": None,
        "session_token": None,
        "account_id": None,
        "account_balance": None,
        "current_speaker": None,
        "just_finished": False,
    }

def run() -> None:
    state = get_initial_state()

    root_memory = ChatMemoryBuffer.from_defaults(token_limit=8000)

    first_run = True
    is_retry = False

    while True:
        if first_run:
            # if this is the first run, start the conversation
            user_msg_str = "Hello"
            first_run = False            
        elif is_retry == True:
            user_msg_str = "That's not right, try again. Pick one agent."
            is_retry = False
        elif state["just_finished"] == True:
            print("Asking the continuation agent to decide what to do next")
            user_msg_str = str(continuation_agent_factory(state).chat("""
                Look at the chat history to date and figure out what the user was originally trying to do.
                They might have had to do some sub-tasks to complete that task, but what we want is the original thing they started out trying to do.                                                                      
                Formulate a sentence as if written by the user that asks to continue that task.
                If it seems like the user really completed their task, output "no_further_task" only.
            """, chat_history=current_history))
            print(f"Continuation agent said {user_msg_str}")
            if user_msg_str == "no_further_task":
                user_msg_str = input(">> ").strip()
            state["just_finished"] = False
        else:
            # any other time, get user input
            user_msg_str = input("> ").strip()            

        current_history = root_memory.get()

        # who should speak next?
        if (state["current_speaker"]):
            print(f"There's already a speaker: {state['current_speaker']}")
            next_speaker = state["current_speaker"]
        else:
            print("No current speaker, asking orchestration agent to decide")
            orchestration_response = orchestration_agent_factory(state).chat(user_msg_str, chat_history=current_history)
            next_speaker = str(orchestration_response).strip()

        #print(f"Next speaker: {next_speaker}")

        if next_speaker == Speaker.STOCK_LOOKUP:
            print("Stock lookup agent selected")
            current_speaker = stock_lookup_agent_factory(state)
            state["current_speaker"] = next_speaker
        elif next_speaker == Speaker.AUTHENTICATE:
            print("Auth agent selected")
            current_speaker = auth_agent_factory(state)
            state["current_speaker"] = next_speaker
        elif next_speaker == Speaker.ACCOUNT_BALANCE:
            print("Account balance agent selected")
            current_speaker = account_balance_agent_factory(state)
            state["current_speaker"] = next_speaker
        elif next_speaker == Speaker.TRANSFER_MONEY:
            print("Transfer money agent selected")
            current_speaker = transfer_money_agent_factory(state)
            state["current_speaker"] = next_speaker
        elif next_speaker == Speaker.CONCIERGE:
            print("Concierge agent selected")
            current_speaker = concierge_agent_factory(state)
        else:
            print("Orchestration agent failed to return a valid speaker; ask it to try again")
            is_retry = True
            continue

        pretty_state = pprint.pformat(state, indent=4)
        #print(f"State: {pretty_state}")

        # chat with the current speaker
        response = current_speaker.chat(user_msg_str, chat_history=current_history)
        print(Fore.MAGENTA + str(response) + Style.RESET_ALL)

        # update chat history
        new_history = current_speaker.memory.get_all()
        root_memory.set(new_history)

if __name__ == "__main__":
    run()
