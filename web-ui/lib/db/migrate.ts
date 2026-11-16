import { config } from "dotenv";
import Database from "better-sqlite3";
import { join } from "path";
import { mkdirSync } from "fs";

config({
  path: ".env.local",
});

const ensureSuggestionColumns = (sqlite: Database.Database) => {
  const tableExists = sqlite
    .prepare(
      "SELECT name FROM sqlite_master WHERE type='table' AND name='suggestion'"
    )
    .get();

  if (!tableExists) {
    return;
  }

  const columns = new Set(
    (
      sqlite.prepare("PRAGMA table_info(suggestion)").all() as Array<{
        name: string;
      }>
    ).map((column) => column.name)
  );

  const additions = [
    { name: "original_text", ddl: "TEXT NOT NULL DEFAULT ''" },
    { name: "suggested_text", ddl: "TEXT NOT NULL DEFAULT ''" },
    { name: "description", ddl: "TEXT NOT NULL DEFAULT ''" },
    { name: "is_resolved", ddl: "INTEGER NOT NULL DEFAULT 0" },
  ];

  for (const addition of additions) {
    if (!columns.has(addition.name)) {
      sqlite.exec(
        `ALTER TABLE suggestion ADD COLUMN ${addition.name} ${addition.ddl}`
      );
      console.log(`➕ Added suggestion.${addition.name}`);
    }
  }
};

const ensureSqliteSchema = (sqlite: Database.Database) => {
  sqlite.exec(`
    CREATE TABLE IF NOT EXISTS user (
      id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
      email TEXT NOT NULL UNIQUE,
      password TEXT,
      created_at INTEGER NOT NULL DEFAULT (unixepoch())
    );

    CREATE TABLE IF NOT EXISTS chat (
      id TEXT PRIMARY KEY,
      created_at INTEGER NOT NULL,
      title TEXT NOT NULL,
      user_id TEXT NOT NULL DEFAULT 'local-user',
      visibility TEXT DEFAULT 'private',
      FOREIGN KEY (user_id) REFERENCES user(id)
    );

    CREATE TABLE IF NOT EXISTS message (
      id TEXT PRIMARY KEY,
      chat_id TEXT NOT NULL,
      role TEXT NOT NULL,
      content TEXT NOT NULL DEFAULT '',
      parts TEXT,
      created_at INTEGER NOT NULL,
      FOREIGN KEY (chat_id) REFERENCES chat(id)
    );

    CREATE TABLE IF NOT EXISTS chat_pdf (
      chat_id TEXT NOT NULL,
      pdf_id TEXT NOT NULL,
      added_at INTEGER NOT NULL DEFAULT (unixepoch()),
      PRIMARY KEY (chat_id, pdf_id),
      FOREIGN KEY (chat_id) REFERENCES chat(id)
    );

    CREATE TABLE IF NOT EXISTS document (
      id TEXT PRIMARY KEY,
      title TEXT NOT NULL,
      content TEXT,
      user_id TEXT NOT NULL,
      kind TEXT,
      created_at INTEGER NOT NULL,
      FOREIGN KEY (user_id) REFERENCES user(id)
    );

    CREATE TABLE IF NOT EXISTS suggestion (
      id TEXT PRIMARY KEY,
      document_id TEXT NOT NULL,
      original_text TEXT NOT NULL DEFAULT '',
      suggested_text TEXT NOT NULL DEFAULT '',
      content TEXT NOT NULL DEFAULT '',
      description TEXT NOT NULL DEFAULT '',
      is_resolved INTEGER NOT NULL DEFAULT 0,
      user_id TEXT NOT NULL,
      document_created_at INTEGER,
      created_at INTEGER NOT NULL,
      FOREIGN KEY (document_id) REFERENCES document(id),
      FOREIGN KEY (user_id) REFERENCES user(id)
    );

    CREATE TABLE IF NOT EXISTS vote (
      chat_id TEXT NOT NULL,
      message_id TEXT NOT NULL,
      is_upvoted INTEGER NOT NULL,
      PRIMARY KEY (chat_id, message_id),
      FOREIGN KEY (chat_id) REFERENCES chat(id)
    );

    CREATE TABLE IF NOT EXISTS stream (
      id TEXT PRIMARY KEY,
      chat_id TEXT NOT NULL,
      content TEXT NOT NULL,
      created_at INTEGER NOT NULL,
      FOREIGN KEY (chat_id) REFERENCES chat(id)
    );
  `);
};

const runMigrate = async () => {
  const dataDir = join(process.cwd(), "data");
  try {
    mkdirSync(dataDir, { recursive: true });
  } catch {
    // Directory may already exist
  }

  const dbPath = join(dataDir, "chat.db");
  console.log(`📁 Using SQLite database at: ${dbPath}`);

  const sqlite = new Database(dbPath);
  const start = Date.now();

  console.log("🔧 Ensuring SQLite schema...");
  ensureSqliteSchema(sqlite);
  ensureSuggestionColumns(sqlite);

  const end = Date.now();
  console.log("✅ Migrations completed in", end - start, "ms");

  sqlite.close();
  process.exit(0);
};

runMigrate().catch((err) => {
  console.error("❌ Migration failed");
  console.error(err);
  process.exit(1);
});
