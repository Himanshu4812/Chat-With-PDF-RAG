-- Additive SQLite migration: suggestion edit fields.
-- Applied idempotently by lib/db/migrate.ts (column existence is checked first),
-- so this file documents the exact DDL for manual restoration.
ALTER TABLE `suggestion` ADD COLUMN `original_text` text NOT NULL DEFAULT '';
--> statement-breakpoint
ALTER TABLE `suggestion` ADD COLUMN `suggested_text` text NOT NULL DEFAULT '';
--> statement-breakpoint
ALTER TABLE `suggestion` ADD COLUMN `description` text NOT NULL DEFAULT '';
--> statement-breakpoint
ALTER TABLE `suggestion` ADD COLUMN `is_resolved` integer NOT NULL DEFAULT 0;
