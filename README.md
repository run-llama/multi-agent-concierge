# Multi-agent concierge system

This repo contains an implementation of a multi-agent concierge system using LlamaIndex's Workflows abstraction. Using this example, you can plug in your own agents and tools to build your own multi-agent system, or hack and extend the underlying code to suit your needs.

In this example, agents are represented by a set name, description, set of tools, and system prompt, which all define how the agent acts and how that agent is selected.

In addition, all agent tools have access to the global state in the workflow, which allows agents to coordinate with each other and share information easily. Tools can also be marked as requiring human confirmation, which will cause the system to ask the user to confirm the tool call before it's sent.

The resulting workflow is rendered automatically using the built-in `draw_all_possible_flows()` and looks like this:

![architecture](./workflow.png)

## Why build this?

Interactive chat bots are by this point a familiar solution to customer service, and agents are a frequent component of chat bot implementations. They provide memory, introspection, tool use and other features necessary for a competent bot.

We have become interested in larger-scale chatbots: ones that can complete dozens of tasks, some of which have dependencies on each other, using hundreds of tools. What would that agent look like? It would have an enormous system prompt and a huge number of tools to choose from, which can be confusing for an agent.

Imagine a bank implementing a system that can:
* Look up the price of a specific stock
* Authenticate a user
* Check your account balance
    * Which requires the user be authenticated
* Transfer money between accounts
    * Which requires the user be authenticated
    * And also that the user checks their account balance first

Each of these top-level tasks has sub-tasks, for instance:
* The stock price lookup might need to look up the stock symbol first
* The user authentication would need to gather a username and a password
* The account balance would need to know which of the user's accounts to check

Coming up with a single primary prompt for all of these tasks and sub-tasks would be very complex. So instead, we designed a multi-agent system with agents responsible for each top-level task, plus a "concierge" agent that can direct the user to the correct agent.

## What we built

We built a system of agents to complete the above tasks. There are four basic "task" agents:
* A stock lookup agent (which takes care of sub-tasks like looking up symbols)
* An authentication agent (which asks for username and password)
* An account balance agent (which takes care of sub-tasks like checking the balance of a specific account)
* A money transfer agent (which takes care of tasks like asking what account to transfer to, and how much)

A **global state** is used, that keeps track of the user and their current state, shared between all the agents. This state is available in any tool call, using the `FunctionToolWithContext` class.

There is also an **orchestration agent**: this agent will interact with the user when no active speaker is set. It will look at the current user state and list of available agents, and decide which agent to route the user to next.

The flow of the the system looks something like this:

![abstract_architecture](./architecture.png)

## Repo Structure

- `main.py` - the main entry point for the application. Sets up the global state and the agent pool, and starts the workflow. See this for a detailed quickstart example of how to use the system.
- `workflow.py` - the workflow definition, including all the agents and tools. This handles orchestration, routing, and human approval.
- `utils.py` - additional utility functions for the workflow, mainly to provide the `FunctionToolWithContext` class.

## The system in action

To get a sense of how this works in practice, here's sample output during an interaction with the system.

At the beginning of the conversation, no active speaker is set, so you get routed to the concierge orchestration agent:

<blockquote>
<span style="color:blue">AGENT >>  Hello! How can I assist you today?</span>
<span style="color:white">USER >> I'd like to make a transfer</span>
<span style="color:green">SYSTEM >>  Transferring to agent Authentication Agent</span>
<span style="color:blue">AGENT >>  To assist with your transfer, I'll need to authenticate you first. Could you please provide your username and password?</span>
</blockquote>

Here, we see the orchestration agent routing to the authentication agent, and then asking for a username and password. This is because the global state does not yet have a username or password.

<blockquote>
<span style="color:white">USER >> username=logan password=abc123</span>
<span style="color:green">SYSTEM >>  Recording username</span>
<span style="color:green">SYSTEM >>  Tool store_username called with {'username': 'logan'} returned None</span>
<span style="color:green">SYSTEM >>  Logging in user logan</span>
<span style="color:green">SYSTEM >>  Tool login called with {'password': 'abc123'} returned Logged in user logan with session token 1234567890. They have an account with id 123 and a balance of $1000.</span>
<span style="color:green">SYSTEM >>  Agent is requesting a transfer. Please hold.</span>
<span style="color:green">SYSTEM >>  Transferring to agent Transfer Money Agent</span>
<span style="color:blue">AGENT >>  You are now authenticated. Please provide the account ID you wish to transfer money to and the amount you'd like to transfer.</span>
</blockquote>

Lots of things are happening here:
- the username and password are stored in the global state
- the authentication agent logs in the user and gathers some account information
- the orchestration agent routes to the transfer money agent
- the transfer money agent requests a transfer amount and account ID

<blockquote>
<span style="color:white">USER >> transfer $123 to account #321</span>
<span style="color:green">SYSTEM >> I need approval for the following tool call:</span>
<span style="color:green">transfer_money</span>
<span style="color:green">{'from_account_id': '123', 'to_account_id': '321', 'amount': 123}</span>
<span style="color:white">Do you approve? (y/n): y</span>
<span style="color:green">SYSTEM >>  Transferring 123 from 123 to account 321</span>
<span style="color:green">SYSTEM >>  Tool transfer_money called with {'from_account_id': '123', 'to_account_id': '321', 'amount': 123} returned Transferred 123 to account 321</span>
<span style="color:blue">AGENT >>  The transfer of $123 to account #321 has been successfully completed. Is there anything else I can help you with?</span>
</blockquote>

Since the transfer tool requires human approval, the orchestration agent asks the user if they approve! If they do, the transfer proceeds.

<blockquote>
<span style="color:white">USER >> I need to lookup the value of a stock</span>
<span style="color:green">SYSTEM >>  Agent is requesting a transfer. Please hold.</span>
<span style="color:green">SYSTEM >>  Transferring to agent Stock Lookup Agent</span>
<span style="color:blue">AGENT >>  Sure, I can help with that. Please provide the name of the company whose stock value you want to look up.</span>
<span style="color:white">USER >> AMD</span>
<span style="color:green">SYSTEM >>  Searching for stock symbol</span>
<span style="color:green">SYSTEM >>  Tool search_for_stock_symbol called with {'company_name': 'AMD'} returned AMD</span>
<span style="color:green">SYSTEM >>  Looking up stock price for AMD</span>
<span style="color:green">SYSTEM >>  Tool lookup_stock_price called with {'stock_symbol': 'AMD'} returned Symbol AMD is currently trading at $100.00</span>
<span style="color:blue">AGENT >>  The current stock price for AMD is $100.00. Is there anything else you would like to know?</span>
</blockquote>

Here, we ask for a stock lookup. The money transfer agent is currently active, so it requests a transfer first, which is then handled by the orchestration agent, and finally the stock lookup agent activated and used to look up the stock price.

<blockquote>
<span style="color:white">USER >> bye</span>
</blockquote>

At any time, the user can end the conversation by saying "bye"/"quit","exit".

## What's next

We think there's some novel stuff in here: coordinating multiple agents "speaking" simultaneously, creating implicit "chains" of agents through natural language instructions, using a "continuation" agent to manage those chains, and using a global state this way. We're excited to see what you do with the patterns we've laid out here.
