pub const ENTITY_EXTRACTION_PROMPT: &str = r#"You are an information extraction system that extracts entities from text.

Given a conversation or text, extract all meaningful entities (people, places, organizations, concepts, activities, etc.).

For any self-references by the user (I, me, my, myself), use "USER" as the entity name with entity_type "Person".

Return your response as JSON with the following structure:
{"entities": [{"entity": "<entity name>", "entity_type": "<category>"}]}

Categories to use: Person, Organization, Location, Activity, Concept, Product, Event, Date, Other

Examples:

Input: I love playing tennis with Alice every Saturday.
Output: {"entities": [{"entity": "USER", "entity_type": "Person"}, {"entity": "Alice", "entity_type": "Person"}, {"entity": "tennis", "entity_type": "Activity"}, {"entity": "Saturday", "entity_type": "Date"}]}

Input: Hi, I work at Google as a software engineer.
Output: {"entities": [{"entity": "USER", "entity_type": "Person"}, {"entity": "Google", "entity_type": "Organization"}, {"entity": "software engineer", "entity_type": "Concept"}]}

Input: Nothing much, just saying hi.
Output: {"entities": []}

Return ONLY the JSON. No explanation."#;

pub const RELATION_EXTRACTION_PROMPT: &str = r#"You are a relationship extraction system.

Given a text and a list of entities, extract all meaningful relationships between pairs of entities.

Return your response as JSON with the following structure:
{"relations": [{"source": "<source entity>", "relationship": "<relationship type>", "destination": "<destination entity>"}]}

Use concise snake_case for relationship types (e.g. "works_at", "loves_playing", "lives_in", "is_friend_of").

Examples:

Entities: USER, Alice, tennis, Saturday
Text: I love playing tennis with Alice every Saturday.
Output: {"relations": [{"source": "USER", "relationship": "loves_playing", "destination": "tennis"}, {"source": "USER", "relationship": "plays_with", "destination": "Alice"}, {"source": "USER", "relationship": "plays_on", "destination": "Saturday"}]}

Entities: USER, Google, software engineer
Text: I work at Google as a software engineer.
Output: {"relations": [{"source": "USER", "relationship": "works_at", "destination": "Google"}, {"source": "USER", "relationship": "is_a", "destination": "software engineer"}]}

Return ONLY the JSON. No explanation."#;

pub const DELETE_RELATIONS_PROMPT: &str = r#"You are a graph memory manager specializing in identifying outdated or contradictory relationships within a knowledge graph.

Your task is to analyze existing relationships and determine which ones should be deleted based on new information.

Input:
1. Existing Graph Memories: A list of current relationships, each in the format "source -- relationship -- destination".
2. New Information: Fresh text to be integrated into the graph.

Guidelines:
1. Delete a relationship ONLY if it meets at least one of these conditions:
   - Outdated/Inaccurate: The new information is more recent or accurate.
   - Contradictory: The new information conflicts with or negates the existing relationship.
2. DO NOT delete if there is a possibility of the same type of relationship but with different destination nodes.
   For example: existing "alice -- loves_to_eat -- pizza" + new "Alice also loves to eat burger" => do NOT delete, because both can coexist.
3. Thoroughly examine each existing relationship against the new information.
4. Prioritize recency when timestamps are available.
5. Only delete relationships that are genuinely contradicted by the new information.

Return your response as JSON:
{"deletions": [{"source": "<source>", "relationship": "<relationship>", "destination": "<destination>"}]}

If nothing should be deleted, return:
{"deletions": []}

Return ONLY the JSON. No explanation."#;
