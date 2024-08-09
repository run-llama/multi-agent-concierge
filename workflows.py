import dotenv
dotenv.load_dotenv()

from llama_index.core.workflow import (
    step, 
    Context, 
    Workflow, 
    Event, 
    StartEvent, 
    StopEvent
)
from llama_index.llms.openai import OpenAI
from llama_index.agent.openai import OpenAIAgent
from llama_index.core.tools import FunctionTool
from enum import Enum
from typing import Optional, List, Callable

class Speaker(str, Enum):
    STOCK_LOOKUP = "stock_lookup"
    AUTHENTICATE = "authenticate"
    ACCOUNT_BALANCE = "account_balance"
    TRANSFER_MONEY = "transfer_money"
    CONCIERGE = "concierge"
    ORCHESTRATOR = "orchestrator"

class InitializeEvent(Event):
    pass

class ConciergeEvent(Event):
    request: Optional[str]
    just_completed: Optional[str]
    need_help: Optional[bool]

class OrchestratorEvent(Event):
    request: str

class StockLookupEvent(Event):
    request: str

class AuthenticateEvent(Event):
    request: str

class AccountBalanceEvent(Event):
    request: str

class TransferMoneyEvent(Event):
    request: str

class ConciergeWorkflow(Workflow):

    @step(pass_context=True)
    async def initialize(self, ctx: Context, ev: InitializeEvent) -> ConciergeEvent:
        ctx.data["user"] = {
            "username": None,
            "session_token": None,
            "account_id": None,
            "account_balance": 0,
        }
        ctx.data["success"] = None
        ctx.data["redirecting"] = None
        ctx.data["overall_request"] = None

        ctx.data["llm"] = llm=OpenAI(model="gpt-4o",temperature=0.4)

        return ConciergeEvent()
  
    @step(pass_context=True)
    async def concierge(self, ctx: Context, ev: ConciergeEvent | StartEvent) -> InitializeEvent | StopEvent:
        # initialize user if not already done
        if ("user" not in ctx.data):
            return InitializeEvent()
        
        # initialize concierge if not already done
        if ("concierge" not in ctx.data):
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
                * checking an account balance
                * transferring money between accounts            
            """)

            ctx.data["concierge"] = OpenAIAgent.from_tools(
                tools=tools,
                llm=ctx.data["llm"],
                system_prompt=system_prompt
            )

        concierge = ctx.data["concierge"]
        if ctx.data["overall_request"] is not None:
            print("There's an overall request in progress, it's ", ctx.data["overall_request"])
            last_request = ctx.data["overall_request"]
            ctx.data["overall_request"] = None
            return OrchestratorEvent(request=last_request)
        elif (ev.just_completed is not None):
            response = concierge.chat(f"FYI, the user has just completed the task: {ev.just_completed}")
        elif (ev.need_help):
            return OrchestratorEvent(request=ev.request)
        else:
            # first time experience
            response = concierge.chat("Hello!")

        print(response)
        user_msg_str = input("> ").strip()
        return OrchestratorEvent(request=user_msg_str)
    
    @step(pass_context=True)
    async def orchestrator(self, ctx: Context, ev: OrchestratorEvent) -> ConciergeEvent | StockLookupEvent | AuthenticateEvent | AccountBalanceEvent | TransferMoneyEvent:

        print(f"Orchestrator received request: {ev.request}")

        def emit_stock_lookup() -> bool:
            """Call this if the user wants to look up a stock price."""
            self.send_event(StockLookupEvent(request=ev.request))
            return True

        def emit_authenticate() -> bool:
            """Call this if the user wants to authenticate"""
            self.send_event(AuthenticateEvent(request=ev.request))
            return True

        def emit_account_balance() -> bool:
            """Call this if the user wants to check an account balance."""
            self.send_event(AccountBalanceEvent(request=ev.request))
            return True

        def emit_transfer_money() -> bool:
            """Call this if the user wants to transfer money."""
            self.send_event(TransferMoneyEvent(request=ev.request))
            return True

        def emit_concierge() -> bool:
            """Call this if the user wants to do something else or you can't figure out what they want to do."""
            self.send_event(ConciergeEvent(request=ev.request))
            return True

        tools = [
            FunctionTool.from_defaults(fn=emit_stock_lookup),
            FunctionTool.from_defaults(fn=emit_authenticate),
            FunctionTool.from_defaults(fn=emit_account_balance),
            FunctionTool.from_defaults(fn=emit_transfer_money),
            FunctionTool.from_defaults(fn=emit_concierge)
        ]
        
        system_prompt = (f"""
            You are on orchestration agent.
            Your job is to decide which agent to run based on the current state of the user and what they've asked to do. You run an agent by calling the appropriate tool for that agent.
                            
            If you did not call any tools, return the string "FAILED" without quotes and nothing else.
        """)

        ctx.data["orchestrator"] = OpenAIAgent.from_tools(
            tools,
            llm=OpenAI(model="gpt-4o",temperature=0.4),
            system_prompt=system_prompt,
        )
        
        orchestrator = ctx.data["orchestrator"]
        response = str(orchestrator.chat(ev.request))

        if response == "FAILED":
            print("Orchestration agent failed to return a valid speaker; try again")
            return OrchestratorEvent(request=ev.request)
        
    @step(pass_context=True)
    async def stock_lookup(self, ctx: Context, ev: StockLookupEvent) -> OrchestratorEvent:

        print(f"Stock lookup received request: {ev.request}")

        if ("stock_lookup_agent" not in ctx.data):
            def lookup_stock_price(stock_symbol: str) -> str:
                """Useful for looking up a stock price."""
                print(f"Looking up stock price for {stock_symbol}")
                return f"Symbol {stock_symbol} is currently trading at $100.00"
            
            def search_for_stock_symbol(str: str) -> str:
                """Useful for searching for a stock symbol given a free-form company name."""
                print("Searching for stock symbol")
                return str.upper()
            
            system_prompt = (f"""
                You are a helpful assistant that is looking up stock prices.
                The user may not know the stock symbol of the company they're interested in,
                so you can help them look it up by the name of the company.
                You can only look up stock symbols given to you by the search_for_stock_symbol tool, don't make them up. Trust the output of the search_for_stock_symbol tool even if it doesn't make sense to you.
                Once you have retrieved a stock price, you *must* call the tool named "done" to signal that you are done. Do this before you respond.
                If the user asks to do anything other than look up a stock symbol or price, call the tool "need_help" to signal some other agent should help.
            """)

            ctx.data["stock_lookup_agent"] = ConciergeAgent(
                name="Stock Lookup Agent",
                parent=self,
                tools=[lookup_stock_price, search_for_stock_symbol],
                context=ctx,
                system_prompt=system_prompt,
                trigger_event=StockLookupEvent
            )

        return ctx.data["stock_lookup_agent"].handle_event(ev)

    @step(pass_context=True)
    async def authenticate(self, ctx: Context, ev: AuthenticateEvent) -> OrchestratorEvent:

        if ("authentication_agent" not in ctx.data):
            def store_username(username: str) -> None:
                """Adds the username to the user state."""
                print("Recording username")
                ctx.data["user"]["username"] = username

            def login(password: str) -> None:
                """Given a password, logs in and stores a session token in the user state."""
                print(f"Logging in {ctx.data['user']['username']}")
                # todo: actually check the password
                session_token = "output_of_login_function_goes_here"
                ctx.data["user"]["session_token"] = session_token
            
            def is_authenticated() -> bool:
                """Checks if the user has a session token."""
                print("Checking if authenticated")
                if ctx.data["user"]["session_token"] is not None:
                    return True

            system_prompt = (f"""
                You are a helpful assistant that is authenticating a user.
                Your task is to get a valid session token stored in the user state.
                To do this, the user must supply you with a username and a valid password. You can ask them to supply these.
                If the user supplies a username and password, call the tool "login" to log them in.
                Once you've called the login tool successfully, call the tool named "done" to signal that you are done. Do this before you respond.
                If the user asks to do anything other than authenticate, call the tool "need_help" to signal some other agent should help.
            """)

            ctx.data["authentication_agent"] = ConciergeAgent(
                name="Authentication Agent",
                parent=self,
                tools=[store_username, login, is_authenticated],
                context=ctx,
                system_prompt=system_prompt,
                trigger_event=AuthenticateEvent
            )

        return ctx.data["authentication_agent"].handle_event(ev)
    
    @step(pass_context=True)
    def account_balance(self, ctx: Context, ev: AccountBalanceEvent) -> OrchestratorEvent:
        
        if("account_balance_agent" not in ctx.data):
            def get_account_id(account_name: str) -> str:
                """Useful for looking up an account ID."""
                print(f"Looking up account ID for {account_name}")
                account_id = "1234567890"
                ctx.data["user"]["account_id"] = account_id
                return f"Account id is {account_id}"
            
            def get_account_balance(account_id: str) -> str:
                """Useful for looking up an account balance."""
                print(f"Looking up account balance for {account_id}")
                ctx.data["user"]["account_balance"] = 1000
                return f"Account {account_id} has a balance of ${ctx.data['user']['account_balance']}"
            
            def is_authenticated() -> bool:
                """Checks if the user is authenticated."""
                print("Account balance agent is checking if authenticated")
                if ctx.data["user"]["session_token"] is not None:
                    return True
                else:
                    return False
                
            def authenticate() -> None:
                """Call this if the user needs to authenticate."""
                print("Account balance agent is authenticating")
                ctx.data["redirecting"] = True
                ctx.data["overall_request"] = "Check account balance"
                self.send_event(AuthenticateEvent(request="Authenticate"))

            system_prompt = (f"""
                You are a helpful assistant that is looking up account balances.
                The user may not know the account ID of the account they're interested in,
                so you can help them look it up by the name of the account.
                The user can only do this if they are authenticated, which you can check with the is_authenticated tool.
                If they aren't authenticated, call the "authenticate" tool to trigger the start of the authentication process; tell them you have done this.
                If they're trying to transfer money, they have to check their account balance first, which you can help with.
                Once you have supplied an account balance, you must call the tool named "done" to signal that you are done. Do this before you respond.
                If the user asks to do anything other than look up an account balance, call the tool "need_help" to signal some other agent should help.
            """)

            ctx.data["account_balance_agent"] = ConciergeAgent(
                name="Account Balance Agent",
                parent=self,
                tools=[get_account_id, get_account_balance, is_authenticated, authenticate],
                context=ctx,
                system_prompt=system_prompt,
                trigger_event=AccountBalanceEvent
            )

        # TODO: this could programmatically check for authentication and emit an event
        # but then the agent wouldn't say anything helpful about what's going on.

        return ctx.data["account_balance_agent"].handle_event(ev)
    
    @step(pass_context=True)
    def transfer_money(self, ctx: Context, ev: TransferMoneyEvent) -> OrchestratorEvent:

        if("transfer_money_agent" not in ctx.data):
            def transfer_money(from_account_id: str, to_account_id: str, amount: int) -> None:
                """Useful for transferring money between accounts."""
                print(f"Transferring {amount} from {from_account_id} account {to_account_id}")
                return f"Transferred {amount} to account {to_account_id}"
            
            def balance_sufficient(account_id: str, amount: int) -> bool:
                """Useful for checking if an account has enough money to transfer."""
                # todo: actually check they've selected the right account ID
                print("Checking if balance is sufficient")
                if ctx.data["user"]['account_balance'] >= amount:
                    return True
                
            def has_balance() -> bool:
                """Useful for checking if an account has a balance."""
                print("Checking if account has a balance")
                if ctx.data["user"]["account_balance"] is not None:
                    return True
            
            def is_authenticated() -> bool:
                """Checks if the user has a session token."""
                print("Transfer money agent is checking if authenticated")
                if ctx.data["user"]["session_token"] is not None:
                    return True
                
            def authenticate() -> None:
                """Call this if the user needs to authenticate."""
                print("Account balance agent is authenticating")
                ctx.data["redirecting"] = True
                ctx.data["overall_request"] = "Check account balance"
                self.send_event(AuthenticateEvent(request="Authenticate"))

            def check_balance() -> None:
                """Call this if the user needs to check their account balance."""
                print("Transfer money agent is checking balance")
                ctx.data["redirecting"] = True
                ctx.data["overall_request"] = "Check account balance"
                self.send_event(AccountBalanceEvent(request="Check balance"))
            
            system_prompt = (f"""
                You are a helpful assistant that transfers money between accounts.
                The user can only do this if they are authenticated, which you can check with the is_authenticated tool.
                If they aren't authenticated, tell them to authenticate first.
                The user must also have looked up their account balance already, which you can check with the has_balance tool.
                If they haven't already, tell them to look up their account balance first.
                Once you have transferred the money, you can call the tool named "done" to signal that you are done. Do this before you respond.
                If the user asks to do anything other than transfer money, call the tool "done" to signal some other agent should help.
            """)

            ctx.data["transfer_money_agent"] = ConciergeAgent(
                name="Transfer Money Agent",
                parent=self,
                tools=[transfer_money, balance_sufficient, has_balance, is_authenticated, authenticate, check_balance],
                context=ctx,
                system_prompt=system_prompt,
                trigger_event=TransferMoneyEvent
            )

        return ctx.data["transfer_money_agent"].handle_event(ev)

class ConciergeAgent():
    name: str
    parent: Workflow
    tools: list[FunctionTool]
    system_prompt: str
    context: Context
    agent: OpenAIAgent
    current_event: Event
    trigger_event: Event

    def __init__(
            self,
            parent: Workflow,
            tools: List[Callable], 
            system_prompt: str, 
            trigger_event: Event,
            context: Context,
            name: str,
        ):
        self.name = name
        self.parent = parent
        self.context = context
        self.system_prompt = system_prompt
        self.context.data["redirecting"] = False
        self.trigger_event = trigger_event

        # set up the tools including the ones everybody gets
        def done() -> None:
            """When you complete your task, call this tool."""
            print(f"{self.name} is complete")

            self.context.data["redirecting"] = True
            parent.send_event(ConciergeEvent(just_completed=self.name))

        def need_help() -> None:
            """If the user asks to do something you don't know how to do, call this."""
            print(f"{self.name} needs help")
            self.context.data["redirecting"] = True
            return ConciergeEvent(request=self.current_event.request,need_help=True)

        self.tools = [
            FunctionTool.from_defaults(fn=done),
            FunctionTool.from_defaults(fn=need_help)
        ]
        for t in tools:
            self.tools.append(FunctionTool.from_defaults(fn=t))

        self.agent = OpenAIAgent.from_tools(
            self.tools,
            llm=self.context.data["llm"],
            system_prompt=system_prompt,
        )

    def handle_event(self, ev: Event):
        self.current_event = ev

        response = str(self.agent.chat(ev.request))
        print(response)

        # if they're sending us elsewhere we're done here
        if self.context.data["redirecting"]:
            self.context.data["redirecting"] = False
            return None

        # otherwise, get some user input and then loop
        user_msg_str = input("> ").strip()
        return self.trigger_event(request=user_msg_str)

async def main():
    c = ConciergeWorkflow(timeout=1200, verbose=True)
    result = await c.run()
    print(result)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
