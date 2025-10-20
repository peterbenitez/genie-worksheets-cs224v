# Quickstart Guide

To create a conversational agent that guides conversations with a user to achieve a certain goal and answers 
users question, you need to provide Genie with:
- Worksheet: A high-level specification for guiding the conversation. Contains the tasks the agent can perform and 
knowledge sources available to the agent.
- Configuration: An entry python file that defines configuration for the agent.

## Creating a Worksheet
Our worksheet design is inspired by the versatility of webforms.
Modern websites contain multiple fields which can be optional, tabs for selection of task, and pop-up windows that 
depend on previous user responses. 
There are two kinds of worksheets: 

- task worksheet: To guide the conversation and achieve goals
- knowledge base worksheet: To answer users questions and assist them in making decisions.

### Task Worksheet
!!! tip "Example"
    Users perform different tasks based on their requirements. A student who is having trouble enrolling in a class 
    would need to fill out a form containing details about the course they want to enroll in, the error message they are
    seeing, and any other additional comments and tasks like Leave of Absence are irrelevant to them. We need to make
    relevant worksheets available.

#### Worksheet Attributes
A task worksheet has the following attributes for a Task worksheet:

- **WS Predicate**: indicates when a task worksheet should be activated based on values of other fields
- **WS Name**: the name of the worksheet. Used by semantic parser.
- **WS Kind**: Should be set to "Task". Defines that this is a Task type worksheet.
- **WS Actions**: Genie provides flexibility to the agent by executing arbitrary Python code to call external APIs, 
change assigned values, or explicitly respond to the user with a given string using built-in actions like say. 
The actions are triggered when all the required parameters for a task are assigned and are defined under WS Actions.

**WS Action examples**
Call external API to fetch information or post

!!! tip "Example for WS Action"
    Perform actions once the user has provided all the required information: If you want to book a restaurant, the agent will call the `book_restaurant` function.
    ```python
    book_restaurant(
        self.restaurant_name, 
        self.date, 
        self.num_people, 
        self.time, 
        self.special_instructions
    )
    ```

!!! tip "Example for Field Action"
    Perform task based on value of a field: Once the user provides the value for `confirm` field, the agent will say "Thank you".
    ```python
    if self.confirm:
        say("Thank you")
    ```

#### Field Attributes

Each task worksheet contains a set of fields. Each field contains the following attributes:

- **Predicate**: indicates when a field should be activated based on values of other fields
- **Kind**: three types of fields can be used: 
    - input: field values that are editable by the user
    - internal: field values that can only be edited by the agent
    - output: set according to the output of executing an API or knowledge base call
- **Types**: Genie allows standard types for each field [TODO: We only use type of guide the semantic parsing, think of them as type hints]: 
    - str: string type
    - bool: boolean type
    - int: integer type
    - float: float type
    - Enum: enumeration type. The values are set in Enum Values cells.
    - list: array of atomic types [TODO: Not implemented yet]
    - confirm: special type of boolean type, that prompts the user to confirm all the field values before performing 
    WS action.
- **Name**: Name of the field. Used by semantic parser.
- **Enum Values**: a set of Enum Values if the type of the field is `Enum`
- **Description**: provides a natural language description of the field.
- **Don't Ask**: a boolean that records the information if the user offers it, but the agent will not ask for it.
    If Don't Ask is false, the agent will ask for the field if it is not assigned, but user can refuse to answer if it
    is not Required.
- **Required**: if the field value is mandatory.
- **Confirmation**: asks for confirmation for the field value if set to TRUE.
- **Actions**: similar to WS Action, is used to execute python code to fetch, modify or post information.

### Knowledge Access Worksheet
Genie Worksheet treats knowledge access as a first-class object.

!!! tip "Example for knowledge access"
    Real-life queries often involve both structured and unstructured accesses.
    For instance, queries "What's the best-rated restaurant with a romantic atmosphere" require access to both the 
    structured "ratings" column and the free text "reviews" column. 

To handle hybrid knowledge bases, Genie adopts the SUQL query language, an SQL extension that integrates search of
unstructured data (Liu et al., 2024d). Genie can be used with other knowledge access systems as well.

For each knowledge base to be included, the developer must create a worksheet with the following attributes:

- **WS Name**: the name of the worksheet. Used by semantic parser.
- **WS Kind**: Should be set to "KB". Defines that this is a Knowledge Base type worksheet.

The attributes for fields should be filled in as following:

- **Kind**: should always be set as `internal` since the user cannot make changes to theses fields. 
    Should also write `primary` if the field is a primary key as: `internal; primary`


## Creating Agents

For creating the agent, we need to load configurations, and add prompts.

### Load the model configuration

```python
from worksheets import Config
import os

# Define path to the prompts
current_dir = os.path.dirname(os.path.realpath(__file__))
prompt_dir = os.path.join(current_dir, "prompts")

# Load config from YAML file
config = Config.load_from_yaml(os.path.join(current_dir, "config.yaml"))
```

You can also define the configuration programmatically:

```python
from worksheets import Config, AzureModelConfig
import os

config = Config(
    semantic_parser=AzureModelConfig(
        model_name="azure/gpt-4o",
    ),
    response_generator=AzureModelConfig(
        model_name="azure/gpt-4o",
    ),
    knowledge_parser=AzureModelConfig(
        model_name="gpt-4o",
    ),
    knowledge_base=AzureModelConfig(
        model_name="azure/gpt-4o",
    ),
)
```

### Define your API functions

```python
from worksheets.agent.config import agent_api
from worksheets.core.worksheet import get_genie_fields_from_ws
from uuid import uuid4

@agent_api("course_detail_to_individual_params", "Get course details")
def course_detail_to_individual_params(course_detail):
    if course_detail.value is None:
        return {}
    course_detail = course_detail.value
    course_detail = {}
    for field in get_genie_fields_from_ws(course_detail):
        course_detail[field.name] = field.value

    return course_detail

@agent_api("courses_to_take_oval", "Final API to enroll into a course")
def courses_to_take_oval(**kwargs):
    return {"success": True, "transaction_id": uuid4()}

@agent_api("is_course_full", "Check if a course is full")
def is_course_full(course_id, **kwargs):
    # Implementation here
    return False
```

### Define your starting prompt

You can load your starting prompt from a template file:

```python
from worksheets.agent.builder import TemplateLoader

starting_prompt = TemplateLoader.load(
    os.path.join(current_dir, "starting_prompt.md"), format="jinja2"
)
```

Or define it inline:

```python
starting_prompt = """Hello! I'm the Course Enrollment Assistant. I can help you with:
- Selecting a course: just say find me programming courses
- Enrolling into a course. 
- Asking me any question related to courses and their requirement criteria.

How can I help you today?"""
```

### Define the Agent

Instead of directly creating an Agent instance, we recommend using AgentBuilder for a more fluent API:

```python
from worksheets import AgentBuilder, SUQLKnowledgeBase, SUQLReActParser

agent = (
    AgentBuilder(
        name="Course Enrollment Assistant",
        description="You are a course enrollment assistant. You can help students with course selection and enrollment.",
        starting_prompt=starting_prompt.render() if hasattr(starting_prompt, 'render') else starting_prompt,
    )
    .with_knowledge_base(
        SUQLKnowledgeBase,
        tables_with_primary_keys={
            "courses": "course_id",
            "ratings": "rating_id",
            "offerings": "course_id",
            "programs": "program_id",
        },
        database_name="course_assistant",
        embedding_server_address="http://127.0.0.1:8509",
        source_file_mapping={
            "course_assistant_general_info.txt": os.path.join(
                current_dir, "course_assistant_general_info.txt"
            )
        },
        postprocessing_fn=postprocess_suql,
        result_postprocessing_fn=None,
        db_username="select_user",
        db_password="select_user",
    )
    .with_parser(
        SUQLReActParser,
        example_path=os.path.join(current_dir, "examples.txt"),
        instruction_path=os.path.join(current_dir, "instructions.txt"),
        table_schema_path=os.path.join(current_dir, "table_schema.txt"),
    )
    .with_gsheet_specification("YOUR_SPREADSHEET_ID_HERE")
    .build(config)
)
```

### Run the conversation loop

```python
import asyncio
from worksheets import conversation_loop

if __name__ == "__main__":
    # Run the conversation loop in the terminal
    asyncio.run(conversation_loop(agent, "output_state_path.json", debug=True))
```

### Add prompts

For each agent you need to create prompts for:
- Semantic parsing: `semantic_parsing.prompt`
- Response generation: `response_generator.prompt`

Place these prompts in the prompt directory that you specify while creating the
agent.

You can copy basic annotated prompts from `experiments/sample_prompts/` 
directory. Make changes where we have `TODO`. You need to provide a few 
guidelines in the prompt that will help the LLM to perform better and some 
examples. Please see `experiments/domain_agents/course_enroll/prompts/` for inspiration!

### Spreadsheet Specification

To create a new agent, you should have a Google Service Account and create a new spreadsheet. 
You can follow the instructions [here](https://cloud.google.com/iam/docs/service-account-overview) to create a Google Service Account.
Share the created spreadsheet with the service account email.

You should save the service_account key as `service_account.json` in the `genie-worksheets/` directory.

Here is a starter worksheet that you can use for your reference: [Starter Worksheet](https://docs.google.com/spreadsheets/d/1ST1ixBogjEEzEhMeb-kVyf-JxGRMjtlRR6z4G2sjyb4/edit?usp=sharing)

Here is a sample spreadsheet for a restaurant agent: [Restaurant Agent](https://docs.google.com/spreadsheets/d/1FXg5VFrdxQlUyld3QmKKL9BN1lLIhAtQTJjCHyNOU_Y/edit?usp=sharing)

Please note that we only use the specification defined in the first sheet of the spreadsheet.

## LLM Config
You should create a `.env` file similar to `.env.example` and fill in the values for the LLM API keys and endpoints.

## Running the Agent (Web Interface)

Create a folder `frontend/`  under `experiments/agents/<agent_name>` and create a `app_*` file.

You can run the agent in a web interface by running:

**NOTE:** You should run the agent in the `frontend` directory to preserve the frontend assets.

For restaurant agent:
```bash
cd experiments/domain_agents/yelpbot/frontend/
chainlit run app_restaurant.py --port 8800
```

Example agents are present in `experiments/agents/` directory. You can use them as a reference to create your own agents.