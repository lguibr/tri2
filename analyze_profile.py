# File: analyze_profile_v2.py
import pstats
from pstats import SortKey

profile_file = "profile_output.prof"
output_file_cumulative = "profile_summary_cumulative.txt"
output_file_tottime = "profile_summary_tottime.txt"
num_lines_to_print = 50  # You can adjust how many lines to show

try:
    # --- Sort by Cumulative Time ---
    print(
        f"Saving top {num_lines_to_print} cumulative time stats to {output_file_cumulative}..."
    )
    with open(output_file_cumulative, "w") as f_cum:
        # Pass the file handle directly as the stream
        stats_cum = pstats.Stats(profile_file, stream=f_cum)
        stats_cum.sort_stats(SortKey.CUMULATIVE).print_stats(num_lines_to_print)
        # 'with open' handles closing/flushing
    print("Done.")

    # --- Sort by Total Time (Internal) ---
    print(
        f"Saving top {num_lines_to_print} total time (tottime) stats to {output_file_tottime}..."
    )
    with open(output_file_tottime, "w") as f_tot:
        # Pass the file handle directly as the stream
        stats_tot = pstats.Stats(profile_file, stream=f_tot)
        stats_tot.sort_stats(SortKey.TIME).print_stats(
            num_lines_to_print
        )  # SortKey.TIME is 'tottime'
        # 'with open' handles closing/flushing
    print("Done.")

    print(
        f"\nAnalysis complete. Check '{output_file_cumulative}' and '{output_file_tottime}'."
    )

except FileNotFoundError:
    print(f"ERROR: Profile file '{profile_file}' not found.")
except Exception as e:
    print(f"An error occurred during profile analysis: {e}")
