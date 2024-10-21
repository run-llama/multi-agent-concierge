import asyncio
from inspect import signature
from pydantic import BaseModel, ConfigDict, Field, create_model
from pydantic.fields import FieldInfo
from typing import Any, Awaitable, Optional, Callable, Type, List, Tuple, Union, cast

from llama_index.core.llms import ChatMessage, LLM
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.program.function_program import get_function_tool
from llama_index.core.tools import (
    BaseTool,
    ToolSelection,
    FunctionTool,
    ToolOutput,
    ToolMetadata,
)
from llama_index.core.workflow import (
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
    Context,
)
from llama_index.core.workflow.events import InputRequiredEvent, HumanResponseEvent
from llama_index.llms.openai import OpenAI

AsyncCallable = Callable[..., Awaitable[Any]]


def create_schema_from_function(
    name: str,
    func: Union[Callable[..., Any], Callable[..., Awaitable[Any]]],
    additional_fields: Optional[
        List[Union[Tuple[str, Type, Any], Tuple[str, Type]]]
    ] = None,
) -> Type[BaseModel]:
    """Create schema from function."""
    fields = {}
    params = signature(func).parameters
    for param_name in params:
        # TODO: Very hacky way to remove the ctx parameter from the signature
        if param_name == "ctx":
            continue

        param_type = params[param_name].annotation
        param_default = params[param_name].default

        if param_type is params[param_name].empty:
            param_type = Any

        if param_default is params[param_name].empty:
            # Required field
            fields[param_name] = (param_type, FieldInfo())
        elif isinstance(param_default, FieldInfo):
            # Field with pydantic.Field as default value
            fields[param_name] = (param_type, param_default)
        else:
            fields[param_name] = (param_type, FieldInfo(default=param_default))

    additional_fields = additional_fields or []
    for field_info in additional_fields:
        if len(field_info) == 3:
            field_info = cast(Tuple[str, Type, Any], field_info)
            field_name, field_type, field_default = field_info
            fields[field_name] = (field_type, FieldInfo(default=field_default))
        elif len(field_info) == 2:
            # Required field has no default value
            field_info = cast(Tuple[str, Type], field_info)
            field_name, field_type = field_info
            fields[field_name] = (field_type, FieldInfo())
        else:
            raise ValueError(
                f"Invalid additional field info: {field_info}. "
                "Must be a tuple of length 2 or 3."
            )

    return create_model(name, **fields)  # type: ignore


class FunctionToolWithContext(FunctionTool):
    """
    A function tool that also includes passing in workflow context.

    Only overrides the call methods to include the context.
    """

    @classmethod
    def from_defaults(
        cls,
        fn: Optional[Callable[..., Any]] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        return_direct: bool = False,
        fn_schema: Optional[Type[BaseModel]] = None,
        async_fn: Optional[AsyncCallable] = None,
        tool_metadata: Optional[ToolMetadata] = None,
    ) -> "FunctionTool":
        if tool_metadata is None:
            fn_to_parse = fn or async_fn
            assert fn_to_parse is not None, "fn or async_fn must be provided."
            name = name or fn_to_parse.__name__
            docstring = fn_to_parse.__doc__

            # TODO: Very hacky way to remove the ctx parameter from the signature
            signature_str = str(signature(fn_to_parse))
            signature_str = signature_str.replace(
                "ctx: llama_index.core.workflow.context.Context, ", ""
            )
            description = description or f"{name}{signature_str}\n{docstring}"
            if fn_schema is None:
                fn_schema = create_schema_from_function(
                    f"{name}", fn_to_parse, additional_fields=None
                )
            tool_metadata = ToolMetadata(
                name=name,
                description=description,
                fn_schema=fn_schema,
                return_direct=return_direct,
            )
        return cls(fn=fn, metadata=tool_metadata, async_fn=async_fn)

    def call(self, ctx: Context, *args: Any, **kwargs: Any) -> ToolOutput:
        """Call."""
        tool_output = self._fn(ctx, *args, **kwargs)
        return ToolOutput(
            content=str(tool_output),
            tool_name=self.metadata.name,
            raw_input={"args": args, "kwargs": kwargs},
            raw_output=tool_output,
        )

    async def acall(self, ctx: Context, *args: Any, **kwargs: Any) -> ToolOutput:
        """Call."""
        tool_output = await self._async_fn(ctx, *args, **kwargs)
        return ToolOutput(
            content=str(tool_output),
            tool_name=self.metadata.name,
            raw_input={"args": args, "kwargs": kwargs},
            raw_output=tool_output,
        )


# ---- Pydantic models for config/llm prediction ----


class AgentConfig(BaseModel):
    """Used to configure an agent."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str
    system_prompt: str | None = None
    tools: list[BaseTool] | None = None
    tools_requiring_human_confirmation: list[str] = Field(default_factory=list)


class TransferToAgent(BaseModel):
    """Used to transfer the user to a specific agent."""

    agent_name: str


class RequestTransfer(BaseModel):
    """Used to signal that either you don't have the tools to complete the task, or you've finished your task and want to transfer to another agent."""

    pass


# ---- Events used to orchestrate the workflow ----


class ActiveSpeakerEvent(Event):
    pass


class OrchestratorEvent(Event):
    pass


class ToolCallEvent(Event):
    tool_call: ToolSelection
    tools: list[BaseTool]


class ToolCallResultEvent(Event):
    chat_message: ChatMessage


class ToolRequestEvent(InputRequiredEvent):
    tool_name: str
    tool_id: str
    tool_kwargs: dict


class ToolApprovedEvent(HumanResponseEvent):
    tool_name: str
    tool_id: str
    tool_kwargs: dict
    approved: bool
    response: str | None = None


class ProgressEvent(Event):
    msg: str


# ---- Workflow ----


class ConciergeAgent(Workflow):
    @step
    async def setup(
        self, ctx: Context, ev: StartEvent
    ) -> ActiveSpeakerEvent | OrchestratorEvent:
        """Sets up the workflow, validates inputs, and stores them in the context."""
        active_speaker = await ctx.get("active_speaker", default="")
        user_msg = ev.get("user_msg")
        agent_configs = ev.get("agent_configs", default=[])
        llm: LLM = ev.get("llm", default=OpenAI(model="gpt-4o", temperature=0.3))
        chat_history = ev.get("chat_history", default=[])
        initial_state = ev.get("initial_state", default={})
        if (
            user_msg is None
            or agent_configs is None
            or llm is None
            or chat_history is None
        ):
            raise ValueError(
                "User message, agent configs, llm, and chat_history are required!"
            )

        if not llm.metadata.is_function_calling_model:
            raise ValueError("LLM must be a function calling model!")

        # store the agent configs in the context
        agent_configs_dict = {ac.name: ac for ac in agent_configs}
        await ctx.set("agent_configs", agent_configs_dict)
        await ctx.set("llm", llm)

        chat_history.append(ChatMessage(role="user", content=user_msg))
        await ctx.set("chat_history", chat_history)

        await ctx.set("user_state", initial_state)

        # if there is an active speaker, we need to transfer forward the user to them
        if active_speaker:
            return ActiveSpeakerEvent()

        # otherwise, we need to decide who the next active speaker is
        return OrchestratorEvent(user_msg=user_msg)

    @step
    async def speak_with_sub_agent(
        self, ctx: Context, ev: ActiveSpeakerEvent
    ) -> ToolCallEvent | ToolRequestEvent | StopEvent:
        """Speaks with the active sub-agent and handles tool calls (if any)."""
        # Setup the agent for the active speaker
        active_speaker = await ctx.get("active_speaker")

        agent_config: AgentConfig = (await ctx.get("agent_configs"))[active_speaker]
        chat_history = await ctx.get("chat_history")
        llm = await ctx.get("llm")

        user_state = await ctx.get("user_state")
        user_state_str = "\n".join([f"{k}: {v}" for k, v in user_state.items()])
        system_prompt = (
            agent_config.system_prompt.strip()
            + f"\n\nHere is the current user state:\n{user_state_str}"
        )

        llm_input = [ChatMessage(role="system", content=system_prompt)] + chat_history

        # inject the request transfer tool into the list of tools
        tools = [get_function_tool(RequestTransfer)] + agent_config.tools

        response = await llm.achat_with_tools(tools, chat_history=llm_input)

        tool_calls: list[ToolSelection] = llm.get_tool_calls_from_response(
            response, error_on_no_tool_call=False
        )
        if len(tool_calls) == 0:
            chat_history.append(response.message)
            await ctx.set("chat_history", chat_history)
            return StopEvent(
                result={
                    "response": response.message.content,
                    "chat_history": chat_history,
                }
            )

        await ctx.set("num_tool_calls", len(tool_calls))

        for tool_call in tool_calls:
            if tool_call.tool_name == "RequestTransfer":
                await ctx.set("active_speaker", None)
                ctx.write_event_to_stream(
                    ProgressEvent(msg="Agent is requesting a transfer. Please hold.")
                )
                return OrchestratorEvent()
            elif tool_call.tool_name in agent_config.tools_requiring_human_confirmation:
                ctx.write_event_to_stream(
                    ToolRequestEvent(
                        prefix=f"Tool {tool_call.tool_name} requires human approval.",
                        tool_name=tool_call.tool_name,
                        tool_kwargs=tool_call.tool_kwargs,
                        tool_id=tool_call.tool_id,
                    )
                )
            else:
                ctx.send_event(
                    ToolCallEvent(tool_call=tool_call, tools=agent_config.tools)
                )

        chat_history.append(response.message)
        await ctx.set("chat_history", chat_history)

    @step
    async def handle_tool_approval(
        self, ctx: Context, ev: ToolApprovedEvent
    ) -> ToolCallEvent | ToolCallResultEvent:
        """Handles the approval or rejection of a tool call."""
        if ev.approved:
            active_speaker = await ctx.get("active_speaker")
            agent_config = (await ctx.get("agent_configs"))[active_speaker]
            return ToolCallEvent(
                tools=agent_config.tools,
                tool_call=ToolSelection(
                    tool_id=ev.tool_id,
                    tool_name=ev.tool_name,
                    tool_kwargs=ev.tool_kwargs,
                ),
            )
        else:
            return ToolCallResultEvent(
                chat_message=ChatMessage(
                    role="tool",
                    content=ev.response
                    or "The tool call was not approved, likely due to a mistake or preconditions not being met.",
                )
            )

    @step(num_workers=4)
    async def handle_tool_call(
        self, ctx: Context, ev: ToolCallEvent
    ) -> ActiveSpeakerEvent:
        """Handles the execution of a tool call."""
        tool_call = ev.tool_call
        tools_by_name = {tool.metadata.get_name(): tool for tool in ev.tools}

        tool_msg = None

        tool = tools_by_name.get(tool_call.tool_name)
        additional_kwargs = {
            "tool_call_id": tool_call.tool_id,
            "name": tool.metadata.get_name(),
        }
        if not tool:
            tool_msg = ChatMessage(
                role="tool",
                content=f"Tool {tool_call.tool_name} does not exist",
                additional_kwargs=additional_kwargs,
            )

        try:
            if isinstance(tool, FunctionToolWithContext):
                tool_output = await tool.acall(ctx, **tool_call.tool_kwargs)
            else:
                tool_output = await tool.acall(**tool_call.tool_kwargs)

            tool_msg = ChatMessage(
                role="tool",
                content=tool_output.content,
                additional_kwargs=additional_kwargs,
            )
        except Exception as e:
            tool_msg = ChatMessage(
                role="tool",
                content=f"Encountered error in tool call: {e}",
                additional_kwargs=additional_kwargs,
            )

        ctx.write_event_to_stream(
            ProgressEvent(
                msg=f"Tool {tool_call.tool_name} called with {tool_call.tool_kwargs} returned {tool_msg.content}"
            )
        )

        return ToolCallResultEvent(chat_message=tool_msg)

    @step
    async def aggregate_tool_results(
        self, ctx: Context, ev: ToolCallResultEvent
    ) -> ActiveSpeakerEvent:
        """Collects the results of all tool calls and updates the chat history."""
        num_tool_calls = await ctx.get("num_tool_calls")
        results = ctx.collect_events(ev, [ToolCallResultEvent] * num_tool_calls)
        if not results:
            return

        chat_history = await ctx.get("chat_history")
        for result in results:
            chat_history.append(result.chat_message)
        await ctx.set("chat_history", chat_history)

        return ActiveSpeakerEvent()

    @step
    async def orchestrator(
        self, ctx: Context, ev: OrchestratorEvent
    ) -> ActiveSpeakerEvent | StopEvent:
        """Decides which agent to run next, if any."""
        agent_configs = await ctx.get("agent_configs")
        chat_history = await ctx.get("chat_history")

        agent_context_str = ""
        for agent_name, agent_config in agent_configs.items():
            agent_context_str += f"{agent_name}: {agent_config.description}\n"

        user_state = await ctx.get("user_state")
        user_state_str = "\n".join([f"{k}: {v}" for k, v in user_state.items()])

        system_prompt = (
            "You are on orchestration agent.\n"
            "Your job is to decide which agent to run based on the current state of the user and what they've asked to do.\n"
            "You do not need to figure out dependencies between agents; the agents will handle that themselves.\n"
            f"Here the the agents you can choose from:\n{agent_context_str}\n\n"
            f"Here is the current user state:\n{user_state_str}\n\n"
            "Please assist the user and transfer them as needed."
        )

        llm_input = [ChatMessage(role="system", content=system_prompt)] + chat_history
        llm = await ctx.get("llm")

        # convert the TransferToAgent pydantic model to a tool
        tools = [get_function_tool(TransferToAgent)]

        response = await llm.achat_with_tools(tools, chat_history=llm_input)
        tool_calls = llm.get_tool_calls_from_response(
            response, error_on_no_tool_call=False
        )

        # if no tool calls were made, the orchestrator probably needs more information
        if len(tool_calls) == 0:
            chat_history.append(response.message)
            return StopEvent(
                result={
                    "response": response.message.content,
                    "chat_history": chat_history,
                }
            )

        tool_call = tool_calls[0]
        selected_agent = tool_call.tool_kwargs["agent_name"]
        await ctx.set("active_speaker", selected_agent)

        ctx.write_event_to_stream(
            ProgressEvent(msg=f"Transferring to agent {selected_agent}")
        )

        return ActiveSpeakerEvent()


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
The user must also have looked up their account balance already, which you can check with the has_balance tool.
If they haven't already, tell them to look up their account balance first and call the "RequestTransfer" tool.
            """,
            tools=get_transfer_money_tools(),
            tools_requiring_human_confirmation=["transfer_money"],
        ),
    ]


async def main():
    """Main function to run the workflow."""

    llm = OpenAI(model="gpt-4o", temperature=0.4)
    memory = ChatMemoryBuffer.from_defaults(llm=llm)
    initial_state = get_initial_state()
    agent_configs = get_agent_configs()
    workflow = ConciergeAgent(timeout=None)

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
                print("SYSTEM >> I need approval for the following tool call:")
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
                print("SYSTEM >> ", event.msg)

        result = await handler
        print("AGENT >> ", result["response"])

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
