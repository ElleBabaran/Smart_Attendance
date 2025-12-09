# TODO: Fix Detection Data in Admin Database

## Tasks
- [x] Update init_database method to add 'date' column to the attendance table
- [x] Modify save_to_database method to save date and time separately
- [x] Update open_admin method to include "Date" column in the treeview and adjust column widths
- [x] Add a "REFRESH" button in the admin panel to reload data from the database
- [x] Add a search bar in the admin panel to filter records by name (last name or first name)
- [x] Optimize camera initialization and detection performance to reduce lag

## Followup Steps
- [x] Test the application to ensure detection data saves with date and time
- [x] Verify that the admin panel displays separate Date and Time columns
- [x] Confirm that the REFRESH button updates the table with latest data
- [x] Test the search bar functionality to filter records as you type
- [x] Test camera opening and ensure reduced lag in live detection
