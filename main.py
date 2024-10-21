import asyncio

from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.tools import BaseTool
from llama_index.core.workflow import Context
from llama_index.llms.openai import OpenAI
from llama_index.utils.workflow import draw_all_possible_flows

from workflow import (
    AgentConfig,
    ConciergeAgent,
    ProgressEvent,
    ToolRequestEvent,
    ToolApprovedEvent,
)
from utils import FunctionToolWithContext


def get_initial_state() -> dict:
    return {
        "username": None,
        "session_token": None,
        "account_id": None,
        "account_balance": None,
    }


def get_stock_lookup_tools() -> list[BaseTool]:
    def lookup_stock_price(ctx: Context, stock_symbol: str) -> str:
        """Useful for looking up a stock price."""
        ctx.write_event_to_stream(
            ProgressEvent(msg=f"Looking up stock price for {stock_symbol}")
        )
        return f"Symbol {stock_symbol} is currently trading at $100.00"

    def search_for_stock_symbol(ctx: Context, company_name: str) -> str:
        """Useful for searching for a stock symbol given a free-form company name."""
        ctx.write_event_to_stream(ProgressEvent(msg="Searching for stock symbol"))
        return company_name.upper()

    return [
        FunctionToolWithContext.from_defaults(fn=lookup_stock_price),
        FunctionToolWithContext.from_defaults(fn=search_for_stock_symbol),
    ]


def get_authentication_tools() -> list[BaseTool]:
    async def store_username(ctx: Context, username: str) -> None:
        """Adds the username to the user state."""
        ctx.write_event_to_stream(ProgressEvent(msg="Recording username"))
        user_state = await ctx.get("user_state")
        user_state["username"] = username
        await ctx.set("user_state", user_state)

    async def login(ctx: Context, password: str) -> str:
        """Given a password, logs in and stores a session token in the user state."""
        user_state = await ctx.get("user_state")
        username = user_state["username"]
        ctx.write_event_to_stream(ProgressEvent(msg=f"Logging in user {username}"))
        # todo: actually check the password
        session_token = "1234567890"
        user_state["session_token"] = session_token
        user_state["account_id"] = "123"
        user_state["account_balance"] = 1000
        await ctx.set("user_state", user_state)

        return f"Logged in user {username} with session token {session_token}. They have an account with id {user_state['account_id']} and a balance of ${user_state['account_balance']}."

    async def is_authenticated(ctx: Context) -> bool:
        """Checks if the user has a session token."""
        ctx.write_event_to_stream(ProgressEvent(msg="Checking if authenticated"))
        user_state = await ctx.get("user_state")
        return user_state["session_token"] is not None

    return [
        FunctionToolWithContext.from_defaults(async_fn=store_username),
        FunctionToolWithContext.from_defaults(async_fn=login),
        FunctionToolWithContext.from_defaults(async_fn=is_authenticated),
    ]


def get_account_balance_tools() -> list[BaseTool]:
    async def get_account_id(ctx: Context, account_name: str) -> str:
        """Useful for looking up an account ID."""
        ctx.write_event_to_stream(
            ProgressEvent(msg=f"Looking up account ID for {account_name}")
        )
        user_state = await ctx.get("user_state")
        account_id = user_state["account_id"]

        return f"Account id is {account_id}"

    async def get_account_balance(ctx: Context, account_id: str) -> str:
        """Useful for looking up an account balance."""
        ctx.write_event_to_stream(
            ProgressEvent(msg=f"Looking up account balance for {account_id}")
        )
        user_state = await ctx.get("user_state")
        account_balance = user_state["account_balance"]

        return f"Account {account_id} has a balance of ${account_balance}"

    async def is_authenticated(ctx: Context) -> bool:
        """Checks if the user has a session token."""
        ctx.write_event_to_stream(ProgressEvent(msg="Checking if authenticated"))
        user_state = await ctx.get("user_state")
        return user_state["session_token"] is not None

    return [
        FunctionToolWithContext.from_defaults(async_fn=get_account_id),
        FunctionToolWithContext.from_defaults(async_fn=get_account_balance),
        FunctionToolWithContext.from_defaults(async_fn=is_authenticated),
    ]


def get_transfer_money_tools() -> list[BaseTool]:
    def transfer_money(
        ctx: Context, from_account_id: str, to_account_id: str, amount: int
    ) -> str:
        """Useful for transferring money between accounts."""
        ctx.write_event_to_stream(
            ProgressEvent(
                msg=f"Transferring {amount} from {from_account_id} to account {to_account_id}"
            )
        )
        return f"Transferred {amount} to account {to_account_id}"

    async def balance_sufficient(ctx: Context, account_id: str, amount: int) -> bool:
        """Useful for checking if an account has enough money to transfer."""
        ctx.write_event_to_stream(
            ProgressEvent(msg="Checking if balance is sufficient")
        )
        user_state = await ctx.get("user_state")
        return user_state["account_balance"] >= amount

    async def has_balance(ctx: Context) -> bool:
        """Useful for checking if an account has a balance."""
        ctx.write_event_to_stream(
            ProgressEvent(msg="Checking if account has a balance")
        )
        user_state = await ctx.get("user_state")
        return (
            user_state["account_balance"] is not None
            and user_state["account_balance"] > 0
        )

    async def is_authenticated(ctx: Context) -> bool:
        """Checks if the user has a session token."""
        ctx.write_event_to_stream(ProgressEvent(msg="Checking if authenticated"))
        user_state = await ctx.get("user_state")
        return user_state["session_token"] is not None

    return [
        FunctionToolWithContext.from_defaults(fn=transfer_money),
        FunctionToolWithContext.from_defaults(async_fn=balance_sufficient),
        FunctionToolWithContext.from_defaults(async_fn=has_balance),
        FunctionToolWithContext.from_defaults(async_fn=is_authenticated),
    ]


def get_agent_configs() -> list[AgentConfig]:
    return [
        AgentConfig(
            name="Stock Lookup Agent",
            description="Looks up stock prices and symbols",
            system_prompt="""
You are a helpful assistant that is looking up stock prices.
The user may not know the stock symbol of the company they're interested in,
so you can help them look it up by the name of the company.
You can only look up stock symbols given to you by the search_for_stock_symbol tool, don't make them up. Trust the output of the search_for_stock_symbol tool even if it doesn't make sense to you.
            """,
            tools=get_stock_lookup_tools(),
        ),
        AgentConfig(
            name="Authentication Agent",
            description="Handles user authentication",
            system_prompt="""
You are a helpful assistant that is authenticating a user.
Your task is to get a valid session token stored in the user state.
To do this, the user must supply you with a username and a valid password. You can ask them to supply these.
If the user supplies a username and password, call the tool "login" to log them in.
Once the user is logged in and authenticated, you can transfer them to another agent.
            """,
            tools=get_authentication_tools(),
        ),
        AgentConfig(
            name="Account Balance Agent",
            description="Checks account balances",
            system_prompt="""
You are a helpful assistant that is looking up account balances.
The user may not know the account ID of the account they're interested in,
so you can help them look it up by the name of the account.
The user can only do this if they are authenticated, which you can check with the is_authenticated tool.
If they aren't authenticated, tell them to authenticate first and call the "RequestTransfer" tool.
If they're trying to transfer money, they have to check their account balance first, which you can help with.
            """,
            tools=get_account_balance_tools(),
        ),
        AgentConfig(
            name="Transfer Money Agent",
            description="Handles money transfers between accounts",
            system_prompt="""
You are a helpful assistant that transfers money between accounts.
The user can only do this if they are authenticated, which you can check with the is_authenticated tool.
If they aren't authenticated, tell them to authenticate first and call the "RequestTransfer" tool.
            """,
            tools=get_transfer_money_tools(),
            tools_requiring_human_confirmation=["transfer_money"],
        ),
    ]


async def main():
    """Main function to run the workflow."""

    from colorama import Fore, Style

    llm = OpenAI(model="gpt-4o", temperature=0.4)
    memory = ChatMemoryBuffer.from_defaults(llm=llm)
    initial_state = get_initial_state()
    agent_configs = get_agent_configs()
    workflow = ConciergeAgent(timeout=None)

    # draw a diagram of the workflow
    draw_all_possible_flows(workflow, filename="workflow.html")

    handler = workflow.run(
        user_msg="Hello!",
        agent_configs=agent_configs,
        llm=llm,
        chat_history=[],
        initial_state=initial_state,
    )

    while True:
        async for event in handler.stream_events():
            if isinstance(event, ToolRequestEvent):
                print(
                    Fore.GREEN
                    + "SYSTEM >> I need approval for the following tool call:"
                    + Style.RESET_ALL
                )
                print(event.tool_name)
                print(event.tool_kwargs)
                print()

                approved = input("Do you approve? (y/n): ")
                if "y" in approved.lower():
                    handler.ctx.send_event(
                        ToolApprovedEvent(
                            tool_id=event.tool_id,
                            tool_name=event.tool_name,
                            tool_kwargs=event.tool_kwargs,
                            approved=True,
                        )
                    )
                else:
                    reason = input("Why not? (reason): ")
                    handler.ctx.send_event(
                        ToolApprovedEvent(
                            tool_name=event.tool_name,
                            tool_id=event.tool_id,
                            tool_kwargs=event.tool_kwargs,
                            approved=False,
                            response=reason,
                        )
                    )
            elif isinstance(event, ProgressEvent):
                print(Fore.GREEN + f"SYSTEM >> {event.msg}" + Style.RESET_ALL)

        result = await handler
        print(Fore.BLUE + f"AGENT >> {result['response']}" + Style.RESET_ALL)

        # update the memory with only the new chat history
        for i, msg in enumerate(result["chat_history"]):
            if i >= len(memory.get()):
                memory.put(msg)

        user_msg = input("USER >> ")
        if user_msg.strip().lower() in ["exit", "quit", "bye"]:
            break

        # pass in the existing context and continue the conversation
        handler = workflow.run(
            ctx=handler.ctx,
            user_msg=user_msg,
            agent_configs=agent_configs,
            llm=llm,
            chat_history=memory.get(),
            initial_state=initial_state,
        )


if __name__ == "__main__":
    asyncio.run(main())
