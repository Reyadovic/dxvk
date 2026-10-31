#pragma once

/*
 * ACU probe: custom addition for DXVK 2.6.0 (not part of upstream DXVK).
 *
 * What it does:
 *  1. Groups every draw call by what it renders into (render target size and
 *     format, depth buffer) and by pixel shader, and counts them.
 *  2. Every few seconds it writes "acu_probe_log.txt" next to the game exe.
 *  3. About once per second it reads "acu_skip.txt" (same folder). That file
 *     lets you drop draw calls live and start an automatic sweep which skips
 *     one pass at a time and measures how much frame time each pass costs.
 *     Results go to "acu_sweep_result.txt".
 *
 * Everything is inside this one header. The only hooks in DXVK are a few
 * one-line calls at the start of the draw/dispatch functions in
 * dxvk_context.cpp.
 */

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#ifndef ACU_PROBE_TEST
  #include "dxvk_include.h"
  #include "dxvk_framebuffer.h"
  #include "dxvk_shader.h"
  #include "../util/util_env.h"
#endif

namespace dxvk {
namespace acu {

  /** Plain description of one draw call (or dispatch). */
  struct DrawInfo {
    uint32_t numColor  = 0;
    uint32_t colorW    = 0;
    uint32_t colorH    = 0;
    uint32_t colorFmt  = 0;
    uint32_t depthW    = 0;
    uint32_t depthH    = 0;
    uint32_t depthFmt  = 0;
    uint32_t shaderId  = 0;      /* pixel shader id, or compute shader id */
    bool     compute   = false;
    uint32_t draws     = 1;
    uint64_t elements  = 0;      /* vertices or indices times instances */
  };


  class Probe {

    enum : uint32_t {
      None      = 0xFFFFFFFFu,
      MaxGroups = 600u,
      MaxClasses = 6000u,
    };

  public:

    static Probe& get() {
      static Probe instance;
      return instance;
    }

    /* Only used by the offline test: choose the folder for the text files. */
    void setDir(const std::string& dir) {
      std::lock_guard<dxvk::mutex> lock(m_mutex);
      m_dir = dir;
      m_dirSet = true;
    }

    /**
     * Called for every draw call and dispatch.
     * Returns true if the call must be skipped.
     */
    bool onDraw(uint32_t frameId, uint64_t nowMs, const DrawInfo& d) {
      std::lock_guard<dxvk::mutex> lock(m_mutex);

      if (!m_started)
        start(nowMs);

      tick(frameId, nowMs);

      uint32_t gi = findGroup(d);
      if (gi == None)
        return false;

      uint32_t ci = findClass(d, gi);
      if (ci == None)
        return false;

      Group& g = m_groups[gi];
      Cls&   c = m_classes[ci];

      if (m_measuring) {
        g.winDraws += d.draws;
        g.winElems += d.elements;
        c.winDraws += d.draws;
        c.winElems += d.elements;
      }

      /* Compute shaders are only ever skipped if explicitly allowed */
      if (d.compute && !m_cfg.skipCompute)
        return false;

      if (m_sweepRunning) {
        const Step& s = m_steps[m_stepPos];

        if (s.baseline)
          return false;

        return m_sweepClasses ? (ci == s.target) : (gi == s.target);
      }

      return g.ruleSkip || c.ruleSkip;
    }

  private:

    struct Group {
      uint32_t    id        = 0;
      std::string label;
      bool        compute   = false;
      bool        ruleSkip  = false;
      uint64_t    winDraws  = 0;
      uint64_t    winElems  = 0;
      uint64_t    prevDraws = 0;
      uint64_t    prevElems = 0;
    };

    struct Cls {
      uint32_t    id        = 0;
      uint32_t    group     = 0;
      std::string label;
      bool        compute   = false;
      bool        ruleSkip  = false;
      uint64_t    winDraws  = 0;
      uint64_t    winElems  = 0;
      uint64_t    prevDraws = 0;
      uint64_t    prevElems = 0;
    };

    struct Config {
      std::vector<std::string> rules;
      int  sweepCount    = 0;
      int  sweepSeconds  = 3;
      int  windowSeconds = 5;
      int  sweepDelay    = -1;
      std::string sweepOnly;
      bool sweepClasses  = false;
      bool skipCompute   = false;
    };

    struct Step {
      bool     baseline = true;
      uint32_t target   = 0;
      double   ms       = 0.0;
    };

    dxvk::mutex m_mutex;

    bool        m_started   = false;
    bool        m_dirSet    = false;
    std::string m_dir;

    std::vector<Group>                      m_groups;
    std::vector<Cls>                        m_classes;
    std::unordered_map<uint64_t, uint32_t>  m_groupMap;
    std::unordered_map<uint64_t, uint32_t>  m_classMap;

    Config      m_cfg;
    std::string m_cfgSig;
    bool        m_cfgSeen   = false;
    uint64_t    m_lastCfgMs = 0;

    bool        m_measuring      = false;
    uint64_t    m_winStartMs     = 0;
    uint64_t    m_measStartMs    = 0;
    uint32_t    m_measStartFrame = 0;

    uint32_t    m_lastFrames     = 0;
    uint64_t    m_lastDurMs      = 0;
    double      m_lastMsPerFrame = 0.0;

    uint64_t          m_startMs          = 0;
    uint64_t          m_nowMs            = 0;
    uint64_t          m_sweepNotBeforeMs = 0;

    bool              m_sweepRunning = false;
    bool              m_sweepClasses = false;
    bool              m_sweepBlock   = false;
    int               m_sweepWanted  = 0;
    std::vector<Step> m_steps;
    size_t            m_stepPos      = 0;


    /* ---------- small helpers ---------- */

    static uint64_t combine(uint64_t h, uint64_t v) {
      h ^= v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
      return h;
    }

    static std::string trim(const std::string& s) {
      size_t b = 0;
      size_t e = s.size();

      while (b < e && (s[b] == ' ' || s[b] == '\t' || s[b] == '\r' || s[b] == '\n'))
        b++;
      while (e > b && (s[e - 1] == ' ' || s[e - 1] == '\t' || s[e - 1] == '\r' || s[e - 1] == '\n'))
        e--;

      return s.substr(b, e - b);
    }

    static bool allDigits(const std::string& s) {
      if (s.empty())
        return false;
      for (char c : s) {
        if (c < '0' || c > '9')
          return false;
      }
      return true;
    }

    static std::string hex8(uint32_t v) {
      char buf[16];
      std::snprintf(buf, sizeof(buf), "%08x", v);
      return std::string(buf);
    }

    static std::string fmtCount(double v) {
      char buf[32];
      if (v >= 1.0e6)
        std::snprintf(buf, sizeof(buf), "%.2fM", v / 1.0e6);
      else if (v >= 1.0e3)
        std::snprintf(buf, sizeof(buf), "%.1fK", v / 1.0e3);
      else
        std::snprintf(buf, sizeof(buf), "%.1f", v);
      return std::string(buf);
    }

    std::string controlPath() const { return m_dir + "acu_skip.txt"; }
    std::string logPath()     const { return m_dir + "acu_probe_log.txt"; }
    std::string resultPath()  const { return m_dir + "acu_sweep_result.txt"; }

    static bool readFile(const std::string& path, std::string& out) {
      std::ifstream f(path.c_str(), std::ios::in | std::ios::binary);
      if (!f.good())
        return false;
      out.assign((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
      return true;
    }

    static void writeFile(const std::string& path, const std::string& text) {
      std::ofstream f(path.c_str(), std::ios::out | std::ios::trunc | std::ios::binary);
      if (f.good())
        f << text;
    }


    /* ---------- startup and config ---------- */

    void start(uint64_t nowMs) {
      m_started = true;
      m_startMs = nowMs;
      m_nowMs   = nowMs;

      if (!m_dirSet) {
        m_dir.clear();
#ifndef ACU_PROBE_TEST
        std::string exe = env::getExePath();
        size_t slash = exe.find_last_of("\\/");
        if (slash != std::string::npos)
          m_dir = exe.substr(0, slash + 1);
#endif
        m_dirSet = true;
      }

      std::string existing;
      if (!readFile(controlPath(), existing)) {
        writeFile(controlPath(),
          "// ACU probe control file. Saved changes are picked up within about a second.\r\n"
          "// Lines starting with // are comments.\r\n"
          "//\r\n"
          "// AUTOMATIC TEST: set sweep=40, save, then stand still somewhere busy.\r\n"
          "// The picture will look broken for a couple of minutes. The result is written\r\n"
          "// to acu_sweep_result.txt. Set it back to sweep=0 afterwards.\r\n"
          "sweep=0\r\n"
          "//\r\n"
          "// TIP: to avoid alt-tabbing, set sweep=40 and sweepDelay=300 BEFORE starting the\r\n"
          "// game. The test then starts by itself 300 seconds after launch, so be standing\r\n"
          "// somewhere busy by then. (Without sweepDelay, a sweep left on from an earlier\r\n"
          "// session will not start by itself.)\r\n"
          "//\r\n"
          "// Optional settings:\r\n"
          "// sweepMode=groups      (groups = whole passes, classes = single pixel shaders)\r\n"
          "// sweepSeconds=3\r\n"
          "// sweepDelay=300        (seconds to wait before the test starts, minimum 30 at launch)\r\n"
          "// sweepOnly=C5 1280x720  (only test passes/shaders whose name contains this text)\r\n"
          "// windowSeconds=5\r\n"
          "// skipCompute=0         (1 = compute shaders may be skipped too)\r\n"
          "//\r\n"
          "// MANUAL SKIPPING: one rule per line, copied from acu_probe_log.txt. Examples:\r\n"
          "// skip g12\r\n"
          "// skip ps:1a2b3c4d\r\n"
          "// skip C0 | D 2048x2048 f126\r\n");
      }

      m_winStartMs = nowMs;
      m_lastCfgMs  = nowMs - 1000u;
    }

    void parseConfig(const std::string& raw, Config& cfg) const {
      size_t pos = 0;

      while (pos <= raw.size()) {
        size_t end = raw.find('\n', pos);
        if (end == std::string::npos)
          end = raw.size();

        std::string line = trim(raw.substr(pos, end - pos));
        pos = end + 1;

        if (line.empty() || line.compare(0, 2, "//") == 0)
          continue;

        if (line.compare(0, 5, "skip ") == 0) {
          std::string pat = trim(line.substr(5));
          if (!pat.empty())
            cfg.rules.push_back(pat);
          continue;
        }

        size_t eq = line.find('=');
        if (eq == std::string::npos)
          continue;

        std::string key = trim(line.substr(0, eq));
        std::string val = trim(line.substr(eq + 1));

        if (key == "sweep")
          cfg.sweepCount = std::max(0, std::min(80, std::atoi(val.c_str())));
        else if (key == "sweepSeconds")
          cfg.sweepSeconds = std::max(1, std::min(30, std::atoi(val.c_str())));
        else if (key == "sweepOnly")
          cfg.sweepOnly = val;
        else if (key == "sweepDelay")
          cfg.sweepDelay = std::max(0, std::min(3600, std::atoi(val.c_str())));
        else if (key == "windowSeconds")
          cfg.windowSeconds = std::max(1, std::min(60, std::atoi(val.c_str())));
        else if (key == "sweepMode")
          cfg.sweepClasses = (val == "classes");
        else if (key == "skipCompute")
          cfg.skipCompute = (std::atoi(val.c_str()) != 0);
      }
    }

    bool ruleMatches(const std::string& rule, char kind, uint32_t id, const std::string& label) const {
      if (rule.size() >= 2 && (rule[0] == 'g' || rule[0] == 'c') && allDigits(rule.substr(1)))
        return rule[0] == kind && uint32_t(std::atoi(rule.c_str() + 1)) == id;

      return label.find(rule) != std::string::npos;
    }

    bool anyRule(char kind, uint32_t id, const std::string& label) const {
      for (const auto& r : m_cfg.rules) {
        if (ruleMatches(r, kind, id, label))
          return true;
      }
      return false;
    }

    void applyRules() {
      for (auto& g : m_groups)
        g.ruleSkip = anyRule('g', g.id, g.label);
      for (auto& c : m_classes)
        c.ruleSkip = anyRule('c', c.id, c.label);
    }

    void reloadConfig(uint64_t nowMs) {
      Config cfg;
      std::string raw;

      if (readFile(controlPath(), raw))
        parseConfig(raw, cfg);

      const bool first = !m_cfgSeen;
      m_cfgSeen = true;

      if (cfg.sweepCount == 0) {
        m_sweepBlock  = false;
        m_sweepWanted = 0;

        if (m_sweepRunning)
          abortSweep(nowMs);
      } else if (!m_sweepBlock && !m_sweepRunning && m_sweepWanted == 0) {
        if (first && cfg.sweepDelay < 30) {
          /* A sweep left switched on from an earlier session must not start
             by itself, unless a sweepDelay of at least 30 seconds is given. */
          m_sweepBlock = true;
        } else {
          int delay = cfg.sweepDelay >= 0 ? cfg.sweepDelay : 5;
          m_sweepWanted      = cfg.sweepCount;
          m_sweepBlock       = true;
          m_sweepNotBeforeMs = (first ? m_startMs : nowMs) + uint64_t(delay) * 1000u;
        }
      }

      std::string sig;
      for (const auto& r : cfg.rules) {
        sig += r;
        sig += '\n';
      }
      sig += "|" + std::to_string(cfg.sweepSeconds) + "|" + std::to_string(cfg.windowSeconds)
          +  "|" + std::to_string(int(cfg.sweepClasses)) + "|" + std::to_string(int(cfg.skipCompute));

      if (sig != m_cfgSig) {
        m_cfgSig = sig;
        m_cfg = cfg;
        applyRules();
        restartWindow(nowMs);
      } else {
        m_cfg.sweepCount = cfg.sweepCount;
        m_cfg.sweepOnly  = cfg.sweepOnly;
      }
    }


    /* ---------- groups and classes ---------- */

    static uint64_t hashGroup(const DrawInfo& d) {
      uint64_t h = 0x1234567u;
      h = combine(h, d.compute ? 1u : 0u);
      if (d.compute)
        return h;
      h = combine(h, d.numColor);
      h = combine(h, d.colorW);
      h = combine(h, d.colorH);
      h = combine(h, d.colorFmt);
      h = combine(h, d.depthW);
      h = combine(h, d.depthH);
      h = combine(h, d.depthFmt);
      return h;
    }

    static std::string groupLabel(const DrawInfo& d) {
      if (d.compute)
        return "COMPUTE";

      char buf[160];
      std::string s;

      if (d.numColor) {
        std::snprintf(buf, sizeof(buf), "C%u %ux%u f%u", d.numColor, d.colorW, d.colorH, d.colorFmt);
        s = buf;
      } else {
        s = "C0";
      }

      s += " | ";

      if (d.depthW) {
        std::snprintf(buf, sizeof(buf), "D %ux%u f%u", d.depthW, d.depthH, d.depthFmt);
        s += buf;
      } else {
        s += "noD";
      }

      return s;
    }

    uint32_t findGroup(const DrawInfo& d) {
      uint64_t key = hashGroup(d);
      auto it = m_groupMap.find(key);

      if (it != m_groupMap.end())
        return it->second;

      if (m_groups.size() >= MaxGroups)
        return None;

      Group g;
      g.id       = uint32_t(m_groups.size()) + 1u;
      g.label    = groupLabel(d);
      g.compute  = d.compute;
      g.ruleSkip = anyRule('g', g.id, g.label);

      uint32_t idx = uint32_t(m_groups.size());
      m_groups.push_back(g);
      m_groupMap.emplace(key, idx);
      return idx;
    }

    uint32_t findClass(const DrawInfo& d, uint32_t gi) {
      uint64_t key = combine(hashGroup(d), uint64_t(d.shaderId) + 7u);
      auto it = m_classMap.find(key);

      if (it != m_classMap.end())
        return it->second;

      if (m_classes.size() >= MaxClasses)
        return None;

      Cls c;
      c.id      = uint32_t(m_classes.size()) + 1u;
      c.group   = gi;
      c.compute = d.compute;
      c.label   = m_groups[gi].label + " | " + (d.compute ? "cs:" : "ps:")
                + (d.shaderId ? hex8(d.shaderId) : std::string("none"));
      c.ruleSkip = anyRule('c', c.id, c.label) || m_groups[gi].ruleSkip;

      uint32_t idx = uint32_t(m_classes.size());
      m_classes.push_back(c);
      m_classMap.emplace(key, idx);
      return idx;
    }


    /* ---------- measurement windows ---------- */

    uint64_t windowMs() const {
      int secs = m_sweepRunning ? m_cfg.sweepSeconds : m_cfg.windowSeconds;
      return uint64_t(secs) * 1000u;
    }

    void restartWindow(uint64_t nowMs) {
      m_measuring  = false;
      m_winStartMs = nowMs;
    }

    void tick(uint32_t frameId, uint64_t nowMs) {
      m_nowMs = nowMs;

      if (nowMs - m_lastCfgMs >= 1000u) {
        m_lastCfgMs = nowMs;
        reloadConfig(nowMs);
      }

      if (!m_measuring) {
        /* Let things settle for half a second after any change */
        if (nowMs - m_winStartMs >= 500u) {
          m_measuring      = true;
          m_measStartMs    = nowMs;
          m_measStartFrame = frameId;

          for (auto& g : m_groups) {
            g.winDraws = 0;
            g.winElems = 0;
          }
          for (auto& c : m_classes) {
            c.winDraws = 0;
            c.winElems = 0;
          }
        }
        return;
      }

      uint64_t len    = windowMs();
      uint64_t dur    = nowMs - m_measStartMs;
      uint32_t frames = frameId - m_measStartFrame;

      if (dur >= len && frames >= 6u)
        endWindow(frames, dur, nowMs);
      else if (dur >= len * 3u)
        restartWindow(nowMs);
    }

    void endWindow(uint32_t frames, uint64_t durMs, uint64_t nowMs) {
      m_lastFrames     = frames;
      m_lastDurMs      = durMs;
      m_lastMsPerFrame = double(durMs) / double(frames);

      for (auto& g : m_groups) {
        g.prevDraws = g.winDraws;
        g.prevElems = g.winElems;
        g.winDraws  = 0;
        g.winElems  = 0;
      }
      for (auto& c : m_classes) {
        c.prevDraws = c.winDraws;
        c.prevElems = c.winElems;
        c.winDraws  = 0;
        c.winElems  = 0;
      }

      writeLog();
      sweepAdvance(m_lastMsPerFrame, nowMs);
      restartWindow(nowMs);
    }


    /* ---------- sweep ---------- */

    std::string labelOf(uint32_t target) const {
      char buf[16];
      if (m_sweepClasses) {
        std::snprintf(buf, sizeof(buf), "c%u ", m_classes[target].id);
        return std::string(buf) + m_classes[target].label;
      }
      std::snprintf(buf, sizeof(buf), "g%u ", m_groups[target].id);
      return std::string(buf) + m_groups[target].label;
    }

    bool matchesOnly(const std::string& label) const {
      return m_cfg.sweepOnly.empty() || label.find(m_cfg.sweepOnly) != std::string::npos;
    }

    void beginSweep(int count) {
      std::vector<uint32_t> idx;
      m_sweepClasses = m_cfg.sweepClasses;

      if (m_sweepClasses) {
        for (uint32_t i = 0; i < m_classes.size(); i++) {
          if (m_classes[i].prevDraws > 0 && (!m_classes[i].compute || m_cfg.skipCompute)
           && matchesOnly(m_classes[i].label))
            idx.push_back(i);
        }
        std::sort(idx.begin(), idx.end(), [this] (uint32_t a, uint32_t b) {
          return m_classes[a].prevDraws > m_classes[b].prevDraws;
        });
      } else {
        for (uint32_t i = 0; i < m_groups.size(); i++) {
          if (m_groups[i].prevDraws > 0 && (!m_groups[i].compute || m_cfg.skipCompute)
           && matchesOnly(m_groups[i].label))
            idx.push_back(i);
        }
        std::sort(idx.begin(), idx.end(), [this] (uint32_t a, uint32_t b) {
          return m_groups[a].prevDraws > m_groups[b].prevDraws;
        });
      }

      if (idx.size() > size_t(count))
        idx.resize(size_t(count));

      if (idx.empty()) {
        writeFile(resultPath(), "ACU SWEEP: nothing to test (no draw calls seen yet).\r\n");
        return;
      }

      m_steps.clear();

      Step base;
      base.baseline = true;
      m_steps.push_back(base);

      for (size_t i = 0; i < idx.size(); i++) {
        Step t;
        t.baseline = false;
        t.target   = idx[i];
        m_steps.push_back(t);

        if ((i + 1) % 4 == 0 && i + 1 != idx.size())
          m_steps.push_back(base);
      }

      m_steps.push_back(base);

      m_stepPos      = 0;
      m_sweepRunning = true;
    }

    void sweepAdvance(double ms, uint64_t nowMs) {
      if (m_sweepRunning) {
        m_steps[m_stepPos].ms = ms;
        m_stepPos++;

        if (m_stepPos >= m_steps.size())
          finishSweep();
        return;
      }

      if (m_sweepWanted > 0 && nowMs >= m_sweepNotBeforeMs) {
        int n = m_sweepWanted;
        m_sweepWanted = 0;
        beginSweep(n);
      }
    }

    void abortSweep(uint64_t nowMs) {
      m_sweepRunning = false;
      m_steps.clear();
      m_stepPos = 0;
      restartWindow(nowMs);
    }

    void finishSweep() {
      struct Res {
        uint32_t target;
        double   saved;
        double   base;
        double   draws;
      };

      std::vector<Res> results;
      double baseSum = 0.0;
      int    baseNum = 0;

      for (size_t p = 0; p < m_steps.size(); p++) {
        if (m_steps[p].baseline) {
          baseSum += m_steps[p].ms;
          baseNum++;
          continue;
        }

        double prevB = -1.0;
        double nextB = -1.0;

        for (size_t q = p; q-- > 0; ) {
          if (m_steps[q].baseline) {
            prevB = m_steps[q].ms;
            break;
          }
        }
        for (size_t q = p + 1; q < m_steps.size(); q++) {
          if (m_steps[q].baseline) {
            nextB = m_steps[q].ms;
            break;
          }
        }

        double base = (prevB > 0.0 && nextB > 0.0) ? (prevB + nextB) * 0.5
                    : (prevB > 0.0 ? prevB : nextB);

        Res r;
        r.target = m_steps[p].target;
        r.draws  = double(m_sweepClasses ? m_classes[r.target].prevDraws : m_groups[r.target].prevDraws)
                 / double(std::max<uint32_t>(m_lastFrames, 1u));
        r.base   = base;
        r.saved  = base - m_steps[p].ms;
        results.push_back(r);
      }

      std::sort(results.begin(), results.end(), [] (const Res& a, const Res& b) {
        return a.saved > b.saved;
      });

      char buf[512];
      std::string out;

      double avgBase = baseNum ? baseSum / baseNum : 0.0;
      std::snprintf(buf, sizeof(buf),
        "ACU SWEEP RESULT\r\n"
        "Each line shows how much frame time was saved while that pass was skipped.\r\n"
        "Bigger = more expensive pass. Differences under about 5%% are noise.\r\n"
        "Baseline (nothing skipped): %.1f ms/frame = %.1f fps\r\n\r\n"
        "saved_ms  saved_%%  draws/f   what\r\n",
        avgBase, avgBase > 0.0 ? 1000.0 / avgBase : 0.0);
      out += buf;

      for (const auto& r : results) {
        double pct = r.base > 0.0 ? 100.0 * r.saved / r.base : 0.0;
        std::snprintf(buf, sizeof(buf), "%8.1f  %7.1f  %7.1f   ", r.saved, pct, r.draws);
        out += buf;
        out += labelOf(r.target);
        out += "\r\n";
      }

      writeFile(resultPath(), out);

      m_sweepRunning = false;
      m_steps.clear();
      m_stepPos = 0;
    }


    /* ---------- log file ---------- */

    std::string activeDescription() const {
      if (m_sweepRunning && m_stepPos < m_steps.size()) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "SWEEP step %u of %u: ",
          unsigned(m_stepPos + 1), unsigned(m_steps.size()));

        const Step& s = m_steps[m_stepPos];
        if (s.baseline)
          return std::string(buf) + "baseline (nothing skipped)";
        return std::string(buf) + "skipping " + labelOf(s.target);
      }

      if (m_cfg.rules.empty())
        return "none";

      std::string s;
      for (const auto& r : m_cfg.rules) {
        if (!s.empty())
          s += " ; ";
        s += r;
      }
      return s;
    }

    void writeLog() {
      char buf[512];
      std::string out;

      double fps = m_lastMsPerFrame > 0.0 ? 1000.0 / m_lastMsPerFrame : 0.0;
      double inv = 1.0 / double(std::max<uint32_t>(m_lastFrames, 1u));

      std::snprintf(buf, sizeof(buf),
        "ACU PROBE   window %.1f s   %u frames   %.1f fps   %.1f ms/frame\r\n",
        double(m_lastDurMs) / 1000.0, unsigned(m_lastFrames), fps, m_lastMsPerFrame);
      out += buf;

      out += "skipped during this window: " + activeDescription() + "\r\n";

      if (m_sweepWanted > 0 && !m_sweepRunning) {
        uint64_t left = m_sweepNotBeforeMs > m_nowMs ? (m_sweepNotBeforeMs - m_nowMs) / 1000u : 0u;
        std::snprintf(buf, sizeof(buf), "sweep armed: starts in about %u s\r\n", unsigned(left));
        out += buf;
      }

      out += "\r\n";

      std::vector<uint32_t> gi;
      for (uint32_t i = 0; i < m_groups.size(); i++) {
        if (m_groups[i].prevDraws > 0)
          gi.push_back(i);
      }
      std::sort(gi.begin(), gi.end(), [this] (uint32_t a, uint32_t b) {
        return m_groups[a].prevDraws > m_groups[b].prevDraws;
      });

      out += "PASSES (render target groups), per frame\r\n";
      out += "id      draws    elements   what\r\n";

      for (size_t i = 0; i < gi.size() && i < 60; i++) {
        const Group& g = m_groups[gi[i]];
        std::snprintf(buf, sizeof(buf), "g%-5u %8.1f  %9s   ",
          g.id, double(g.prevDraws) * inv, fmtCount(double(g.prevElems) * inv).c_str());
        out += buf;
        out += g.label + "\r\n";
      }

      std::vector<uint32_t> ci;
      for (uint32_t i = 0; i < m_classes.size(); i++) {
        if (m_classes[i].prevDraws > 0)
          ci.push_back(i);
      }
      std::sort(ci.begin(), ci.end(), [this] (uint32_t a, uint32_t b) {
        return m_classes[a].prevDraws > m_classes[b].prevDraws;
      });

      out += "\r\nTOP SHADER CLASSES, per frame\r\n";
      out += "id      draws    elements   what\r\n";

      for (size_t i = 0; i < ci.size() && i < 60; i++) {
        const Cls& c = m_classes[ci[i]];
        std::snprintf(buf, sizeof(buf), "c%-5u %8.1f  %9s   ",
          c.id, double(c.prevDraws) * inv, fmtCount(double(c.prevElems) * inv).c_str());
        out += buf;
        out += c.label + "\r\n";
      }

      writeFile(logPath(), out);
    }

  };

}


#ifndef ACU_PROBE_TEST

  inline uint64_t acuNowMs() {
    return uint64_t(std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count());
  }

  inline uint32_t acuShaderId(const Rc<DxvkShader>& shader) {
    return shader != nullptr ? shader->getShaderKey().sha1().dword(0) : 0u;
  }

  inline void acuFillTargets(acu::DrawInfo& info, const DxvkRenderTargets& rt) {
    for (uint32_t i = 0; i < MaxNumRenderTargets; i++) {
      if (rt.color[i].view == nullptr)
        continue;

      if (info.numColor == 0u) {
        VkExtent3D ext = rt.color[i].view->mipLevelExtent(0);
        info.colorW   = ext.width;
        info.colorH   = ext.height;
        info.colorFmt = uint32_t(rt.color[i].view->info().format);
      }

      info.numColor += 1u;
    }

    if (rt.depth.view != nullptr) {
      VkExtent3D ext = rt.depth.view->mipLevelExtent(0);
      info.depthW   = ext.width;
      info.depthH   = ext.height;
      info.depthFmt = uint32_t(rt.depth.view->info().format);
    }
  }

  /** Direct draws. Returns true if the draw should be skipped. */
  template<bool Indexed, typename T>
  inline bool acuSkipDraw(
          uint32_t              frameId,
    const DxvkRenderTargets&    rt,
    const Rc<DxvkShader>&       fs,
          uint32_t              count,
    const T*                    draws) {
    acu::DrawInfo info;
    acuFillTargets(info, rt);
    info.shaderId = acuShaderId(fs);
    info.draws    = count;

    for (uint32_t i = 0; i < count; i++) {
      if constexpr (Indexed)
        info.elements += uint64_t(draws[i].indexCount) * uint64_t(draws[i].instanceCount);
      else
        info.elements += uint64_t(draws[i].vertexCount) * uint64_t(draws[i].instanceCount);
    }

    return acu::Probe::get().onDraw(frameId, acuNowMs(), info);
  }

  /** Indirect draws, where the element count is not known on the CPU. */
  inline bool acuSkipIndirect(
          uint32_t              frameId,
    const DxvkRenderTargets&    rt,
    const Rc<DxvkShader>&       fs,
          uint32_t              count) {
    acu::DrawInfo info;
    acuFillTargets(info, rt);
    info.shaderId = acuShaderId(fs);
    info.draws    = count;

    return acu::Probe::get().onDraw(frameId, acuNowMs(), info);
  }

  /** Compute dispatches. */
  inline bool acuSkipCompute(
          uint32_t              frameId,
    const Rc<DxvkShader>&       cs,
          uint64_t              workgroups) {
    acu::DrawInfo info;
    info.compute  = true;
    info.shaderId = acuShaderId(cs);
    info.elements = workgroups;

    return acu::Probe::get().onDraw(frameId, acuNowMs(), info);
  }

#endif

}
