from typing import Any
from pydantic import BaseModel, ConfigDict, Field

from llama_index.core.llms import ChatMessage, LLM
from llama_index.core.program.function_program import get_function_tool
from llama_index.core.tools import (
    BaseTool,
    ToolSelection,
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

from utils import FunctionToolWithContext


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

DEFAULT_ORCHESTRATOR_PROMPT = (
    "You are on orchestration agent.\n"
    "Your job is to decide which agent to run based on the current state of the user and what they've asked to do.\n"
    "You do not need to figure out dependencies between agents; the agents will handle that themselves.\n"
    "Here the the agents you can choose from:\n{agent_context_str}\n\n"
    "Here is the current user state:\n{user_state_str}\n\n"
    "Please assist the user and transfer them as needed."
)
DEFAULT_TOOL_REJECT_STR = "The tool call was not approved, likely due to a mistake or preconditions not being met."


class ConciergeAgent(Workflow):
    def __init__(
        self,
        orchestrator_prompt: str | None = None,
        default_tool_reject_str: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.orchestrator_prompt = orchestrator_prompt or DEFAULT_ORCHESTRATOR_PROMPT
        self.default_tool_reject_str = (
            default_tool_reject_str or DEFAULT_TOOL_REJECT_STR
        )

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
                    content=ev.response or self.default_tool_reject_str,
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
        system_prompt = self.orchestrator_prompt.format(
            agent_context_str=agent_context_str, user_state_str=user_state_str
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
