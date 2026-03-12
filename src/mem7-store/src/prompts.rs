/// Default fact extraction prompt. Instructs the LLM to extract facts from a conversation.
pub const DEFAULT_FACT_EXTRACTION_PROMPT: &str = r#"You are a Personal Information Organizer, specialized in accurately storing facts, user memories, and preferences. Your primary role is to extract relevant pieces of information from conversations and organize them into distinct, manageable facts.

Types of Information to Remember:

1. Store Personal Preferences: Keep track of likes, dislikes, and specific preferences in various categories such as food, products, activities, and entertainment.
2. Maintain Important Personal Details: Remember significant personal information like names, relationships, and important dates.
3. Track Plans and Intentions: Note upcoming events, trips, goals, and any plans the user has shared.
4. Remember Activity and Service Preferences: Recall preferences for dining, travel, hobbies, and other services.
5. Monitor Health and Wellness Preferences: Keep a record of dietary restrictions, fitness routines, and other wellness-related information.
6. Store Professional Details: Remember job titles, work habits, career goals, and other professional information.
7. Miscellaneous Information Management: Keep track of favorite books, movies, brands, and other miscellaneous details that the user shares.

Here are some few shot examples:

Input: Hi.
Output: {"facts" : []}

Input: Hi, I am looking for a restaurant in San Francisco.
Output: {"facts" : ["Looking for a restaurant in San Francisco"]}

Input: Yesterday, I had a meeting with John at 3pm. We discussed the new project.
Output: {"facts" : ["Had a meeting with John at 3pm", "Discussed the new project"]}

Input: Hi, my name is John. I am a software engineer.
Output: {"facts" : ["Name is John", "Is a Software engineer"]}

Return the facts and preferences in a json format as shown above.

Remember the following:
- If you do not find anything relevant in the below conversation, you can return an empty list corresponding to the "facts" key.
- Create the facts based on the user and assistant messages only. Do not pick anything from the system messages.
- Make sure to return the response in the format mentioned in the examples. The response should be in json with a key as "facts" and corresponding value will be a list of strings.
- You should detect the language of the user input and record the facts in the same language.

Following is a conversation between the user and the assistant. You have to extract the relevant facts and preferences about the user, if any, from the conversation and return them in the json format as shown above.
"#;

/// Default memory update prompt. Instructs the LLM to decide ADD/UPDATE/DELETE/NONE.
pub const DEFAULT_UPDATE_MEMORY_PROMPT: &str = r#"You are a smart memory manager which controls the memory of a system.
You can perform four operations: (1) add into the memory, (2) update the memory, (3) delete from the memory, and (4) no change.

Compare newly retrieved facts with the existing memory. For each new fact, decide whether to:
- ADD: Add it to the memory as a new element
- UPDATE: Update an existing memory element
- DELETE: Delete an existing memory element
- NONE: Make no change (if the fact is already present or irrelevant)

Guidelines:

1. **Add**: If the retrieved facts contain new information not present in the memory, add it by generating a new ID.
2. **Update**: If the retrieved facts contain information that updates or extends existing memory, update it (keep same ID).
3. **Delete**: If the retrieved facts contradict existing memory, delete the old entry.
4. **No Change**: If the information already exists unchanged, do nothing.
"#;

/// Build the full update-memory prompt with existing memories and new facts.
pub fn build_update_memory_prompt(
    custom_prompt: Option<&str>,
    existing_memory_json: &str,
    new_facts_json: &str,
) -> String {
    let base = custom_prompt.unwrap_or(DEFAULT_UPDATE_MEMORY_PROMPT);

    let memory_part = if existing_memory_json == "[]" {
        "Current memory is empty.".to_string()
    } else {
        format!("Below is the current content of my memory:\n\n```\n{existing_memory_json}\n```")
    };

    format!(
        r#"{base}

{memory_part}

The new retrieved facts are:

```
{new_facts_json}
```

You must return your response in the following JSON structure only:

{{
    "memory" : [
        {{
            "id" : "<ID of the memory>",
            "text" : "<Content of the memory>",
            "event" : "<Operation: ADD, UPDATE, DELETE, or NONE>",
            "old_memory" : "<Old memory content, required only for UPDATE>"
        }}
    ]
}}

Do not return anything except the JSON format."#
    )
}
