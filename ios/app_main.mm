// DressiExamples: every example in one app, like the Android one. Pick one,
// watch its streams, read the log, stop anytime. Outputs land in
// Documents/out/<example>/. Launch arguments for unattended runs:
//   --run=<entry>   run one list entry, then exit with its code
//   --run-all       run every entry in turn, exit non-zero if any failed
// and everything after `--` is appended to the example's own arguments.
#import <UIKit/UIKit.h>
#include <spdlog/sinks/callback_sink.h>
#include <spdlog/sinks/stdout_sinks.h>
#include <spdlog/spdlog.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "example_host.h"

namespace {

using dressi_examples::ExampleHost;
using dressi_examples::IViewer;

// One list row: a registry example plus its arguments.
struct Entry {
    std::string label;    // shown in the list; --run= key; out/ folder name
    std::string example;  // registry name
    std::vector<std::string> args;
};

std::string BundleData() {
    return std::string(NSBundle.mainBundle.resourcePath.UTF8String) + "/data";
}

std::string DocumentsDir() {
    return NSSearchPathForDirectoriesInDomains(
                   NSDocumentDirectory, NSUserDomainMask, YES)
            .firstObject.UTF8String;
}

std::vector<Entry> MakeEntries() {
    const std::string data = BundleData();
    const std::string bunny = "--data-dir=" + data + "/bunny";
    const std::string mesh =
            "--mesh=" + data + "/DamagedHelmet/glTF/DamagedHelmet.gltf";
    const std::string env = "--env=" + data + "/suburban_garden_512.exr";
    std::vector<Entry> entries = {
            {"image_fitting", "image_fitting", {}},
            {"texture_optimization", "texture_optimization", {bunny}},
            {"silhouette_optimization_hardsoftras", "silhouette_optimization",
             {bunny, "--technique=hardsoftras"}},
            {"silhouette_optimization_aa", "silhouette_optimization",
             {bunny, "--technique=aa"}},
            {"shape_texture_optimization", "shape_texture_optimization",
             {bunny}},
            // A fixed frame budget: pbr_shading otherwise never terminates
            {"pbr_shading", "pbr_shading", {mesh, env, "--frames=200"}},
            {"pbr_material_optimization", "pbr_material_optimization",
             {mesh, env}},
            {"pbr_envmap_optimization", "pbr_envmap_optimization",
             {mesh, env}},
    };
    for (auto& e : entries) {
        e.args.push_back("--out-dir=" + DocumentsDir() + "/out/" + e.label);
    }
    return entries;
}

dressi_examples::RunFn FindRun(const std::string& name) {
    for (const auto& info : dressi_examples::GetExampleRegistry()) {
        if (name == info.name) {
            return info.run;
        }
    }
    return nullptr;
}

// Latest frame of every stream, shared between the runner thread and the UI.
class Streams {
public:
    struct Frame {
        uint32_t w = 0, h = 0;
        std::vector<uint8_t> rgba;
    };

    static Streams& instance() {
        static Streams s;
        return s;
    }

    void reset() {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_titles.clear();
        m_frames.clear();
    }

    size_t add(const std::string& title) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_titles.push_back(title);
        m_frames.emplace_back();
        return m_titles.size() - 1;
    }

    void setFrame(size_t id, const dressi::CpuImage& img) {
        Frame f;
        f.w = img.width;
        f.h = img.height;
        f.rgba.resize(size_t(f.w) * f.h * 4);
        const auto to8 = [](float v) {
            return uint8_t(std::clamp(v, 0.f, 1.f) * 255.f + 0.5f);
        };
        const uint32_t c = img.channels;
        for (size_t i = 0; i < size_t(f.w) * f.h; i++) {
            const float* px = img.data.data() + i * c;
            f.rgba[i * 4 + 0] = to8(px[0]);
            f.rgba[i * 4 + 1] = to8(px[c >= 3 ? 1 : 0]);
            f.rgba[i * 4 + 2] = to8(px[c >= 3 ? 2 : 0]);
            f.rgba[i * 4 + 3] = 255;
        }
        std::lock_guard<std::mutex> lock(m_mutex);
        m_frames[id] = std::move(f);
    }

    std::vector<std::string> titles() {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_titles;
    }

    // Stream `index`, or every stream stacked vertically when index < 0.
    Frame frame(int index) {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (index >= 0) {
            return size_t(index) < m_frames.size() ? m_frames[size_t(index)]
                                                   : Frame{};
        }
        Frame all;
        for (const auto& f : m_frames) {
            all.w = std::max(all.w, f.w);
            all.h += f.h;
        }
        all.rgba.assign(size_t(all.w) * all.h * 4, 0);
        size_t y0 = 0;
        for (const auto& f : m_frames) {
            for (uint32_t y = 0; y < f.h; y++) {
                std::copy_n(f.rgba.data() + size_t(y) * f.w * 4, size_t(f.w) * 4,
                            all.rgba.data() + (y0 + y) * all.w * 4);
            }
            y0 += f.h;
        }
        return all;
    }

private:
    std::mutex m_mutex;
    std::vector<std::string> m_titles;
    std::vector<Frame> m_frames;
};

class IosViewer : public IViewer {
public:
    explicit IosViewer(const std::string& title)
        : m_id(Streams::instance().add(title)) {}
    bool valid() const override { return true; }
    bool update(const dressi::CpuImage& img) override {
        Streams::instance().setFrame(m_id, img);
        return true;
    }
    void setTitle(const std::string&) override {}
    void setPosition(int, int) override {}

private:
    size_t m_id;
};

class IosHost : public ExampleHost {
public:
    explicit IosHost(const std::atomic<bool>& cancel) : m_cancel(cancel) {}
    std::unique_ptr<IViewer> makeViewer(uint32_t, uint32_t,
                                        const std::string& title) override {
        return std::make_unique<IosViewer>(title);
    }
    bool cancelled() const override {
        return m_cancel.load(std::memory_order_relaxed);
    }

private:
    const std::atomic<bool>& m_cancel;
};

std::atomic<bool> g_cancel{false};
std::atomic<bool> g_running{false};

// Log lines for the UI, polled by the run screen.
std::mutex g_log_mutex;
std::string g_log;

void InstallLogSinks() {
    auto ui_sink = std::make_shared<spdlog::sinks::callback_sink_mt>(
            [](const spdlog::details::log_msg& msg) {
                std::lock_guard<std::mutex> lock(g_log_mutex);
                g_log.append(msg.payload.data(), msg.payload.size());
                g_log.push_back('\n');
                constexpr size_t kMaxLog = 32 << 10;
                if (g_log.size() > kMaxLog) {
                    g_log.erase(0, g_log.size() - kMaxLog);
                }
            });
    auto logger = std::make_shared<spdlog::logger>(
            "dressi",
            spdlog::sinks_init_list{
                    ui_sink,
                    std::make_shared<spdlog::sinks::stdout_sink_mt>()});
    logger->flush_on(spdlog::level::info);
    spdlog::set_default_logger(logger);
}

int RunEntry(const Entry& entry, const std::vector<std::string>& extra) {
    Streams::instance().reset();
    std::vector<std::string> args = entry.args;
    args.insert(args.end(), extra.begin(), extra.end());
    IosHost host(g_cancel);
    const int rc = FindRun(entry.example)(args, host);
    spdlog::info("[dressi-ios] {} exit={}", entry.label, rc);
    return rc;
}

// Examples run on a thread with a desktop-sized stack: secondary threads
// default to 512 KiB, too little for glslang on the larger fused shaders.
void StartRunner(void (^body)(void)) {
    NSThread* thread = [[NSThread alloc] initWithBlock:body];
    thread.stackSize = 64 << 20;
    [thread start];
}

}  // namespace

// ------------------------------- Run screen --------------------------------
@interface RunViewController : UIViewController
- (instancetype)initWithEntry:(const Entry&)entry;
@end

@implementation RunViewController {
    Entry _entry;
    UIImageView* _image;
    UISegmentedControl* _streams;
    UITextView* _log;
    NSTimer* _timer;
    size_t _shown_streams;
}

- (instancetype)initWithEntry:(const Entry&)entry {
    if ((self = [super initWithNibName:nil bundle:nil])) {
        _entry = entry;
    }
    return self;
}

- (void)viewDidLoad {
    [super viewDidLoad];
    self.title = @(_entry.label.c_str());
    self.view.backgroundColor = UIColor.systemBackgroundColor;
    self.navigationItem.rightBarButtonItem = [[UIBarButtonItem alloc]
            initWithTitle:@"Stop"
                    style:UIBarButtonItemStylePlain
                   target:self
                   action:@selector(stop)];

    _image = [UIImageView new];
    _image.contentMode = UIViewContentModeScaleAspectFit;
    _image.backgroundColor = UIColor.blackColor;
    _image.layer.magnificationFilter = kCAFilterNearest;
    _streams = [UISegmentedControl new];
    _log = [UITextView new];
    _log.editable = NO;
    _log.font = [UIFont monospacedSystemFontOfSize:10
                                            weight:UIFontWeightRegular];

    UIStackView* stack = [[UIStackView alloc]
            initWithArrangedSubviews:@[ _image, _streams, _log ]];
    stack.axis = UILayoutConstraintAxisVertical;
    stack.spacing = 4;
    stack.translatesAutoresizingMaskIntoConstraints = NO;
    [self.view addSubview:stack];
    UILayoutGuide* g = self.view.safeAreaLayoutGuide;
    [NSLayoutConstraint activateConstraints:@[
        [stack.topAnchor constraintEqualToAnchor:g.topAnchor],
        [stack.bottomAnchor constraintEqualToAnchor:g.bottomAnchor],
        [stack.leadingAnchor constraintEqualToAnchor:g.leadingAnchor],
        [stack.trailingAnchor constraintEqualToAnchor:g.trailingAnchor],
        [_image.heightAnchor constraintEqualToAnchor:g.heightAnchor
                                          multiplier:0.6],
    ]];

    {
        std::lock_guard<std::mutex> lock(g_log_mutex);
        g_log.clear();
    }
    g_cancel = false;
    g_running = true;
    UIApplication.sharedApplication.idleTimerDisabled = YES;
    const Entry entry = _entry;
    StartRunner(^{
        RunEntry(entry, {});
        g_running = false;
    });
    _timer = [NSTimer scheduledTimerWithTimeInterval:1.0 / 15
                                              target:self
                                            selector:@selector(refresh)
                                            userInfo:nil
                                             repeats:YES];
}

- (void)stop {
    g_cancel = true;
}

- (void)viewWillDisappear:(BOOL)animated {
    [super viewWillDisappear:animated];
    if (self.isMovingFromParentViewController) {
        g_cancel = true;
        [_timer invalidate];
        UIApplication.sharedApplication.idleTimerDisabled = NO;
    }
}

- (void)refresh {
    const auto titles = Streams::instance().titles();
    if (titles.size() != _shown_streams) {
        _shown_streams = titles.size();
        [_streams removeAllSegments];
        for (size_t i = 0; i < titles.size(); i++) {
            [_streams insertSegmentWithTitle:@(titles[i].c_str())
                                     atIndex:i
                                    animated:NO];
        }
        if (titles.size() >= 2) {
            [_streams insertSegmentWithTitle:@"all"
                                     atIndex:titles.size()
                                    animated:NO];
        }
        _streams.selectedSegmentIndex = _streams.numberOfSegments - 1;
    }
    const NSInteger sel = _streams.selectedSegmentIndex;
    const int stream = (titles.size() >= 2 && sel == NSInteger(titles.size()))
                               ? -1
                               : int(sel);
    auto frame = Streams::instance().frame(stream);
    if (frame.w != 0 && frame.h != 0) {
        NSData* bytes = [NSData dataWithBytes:frame.rgba.data()
                                       length:frame.rgba.size()];
        CGDataProviderRef provider =
                CGDataProviderCreateWithCFData((__bridge CFDataRef)bytes);
        CGColorSpaceRef cs = CGColorSpaceCreateDeviceRGB();
        CGImageRef cg = CGImageCreate(
                frame.w, frame.h, 8, 32, size_t(frame.w) * 4, cs,
                kCGBitmapByteOrderDefault | kCGImageAlphaNoneSkipLast,
                provider, nullptr, false, kCGRenderingIntentDefault);
        _image.image = [UIImage imageWithCGImage:cg];
        CGImageRelease(cg);
        CGColorSpaceRelease(cs);
        CGDataProviderRelease(provider);
    }
    std::string text;
    {
        std::lock_guard<std::mutex> lock(g_log_mutex);
        text = g_log;
    }
    if (_log.text.length != text.size()) {
        _log.text = @(text.c_str());
        [_log scrollRangeToVisible:NSMakeRange(_log.text.length, 0)];
    }
    if (!g_running) {
        self.navigationItem.rightBarButtonItem.enabled = NO;
    }
}

@end

// ------------------------------- Example list ------------------------------
@interface ExampleListController : UITableViewController
@end

@implementation ExampleListController {
    std::vector<Entry> _entries;
}

- (void)viewDidLoad {
    [super viewDidLoad];
    self.title = @"dressi";
    _entries = MakeEntries();
    [self.tableView registerClass:UITableViewCell.class
           forCellReuseIdentifier:@"cell"];
}

- (NSInteger)tableView:(UITableView*)tv numberOfRowsInSection:(NSInteger)s {
    return NSInteger(_entries.size());
}

- (UITableViewCell*)tableView:(UITableView*)tv
        cellForRowAtIndexPath:(NSIndexPath*)path {
    UITableViewCell* cell = [tv dequeueReusableCellWithIdentifier:@"cell"
                                                     forIndexPath:path];
    cell.textLabel.text = @(_entries[size_t(path.row)].label.c_str());
    cell.textLabel.adjustsFontSizeToFitWidth = YES;
    return cell;
}

- (void)tableView:(UITableView*)tv didSelectRowAtIndexPath:(NSIndexPath*)path {
    [tv deselectRowAtIndexPath:path animated:YES];
    if (g_running) {
        return;  // one run at a time; the previous one is still stopping
    }
    [self.navigationController
            pushViewController:[[RunViewController alloc]
                                       initWithEntry:_entries[size_t(path.row)]]
                      animated:YES];
}

@end

// ------------------------------- Application -------------------------------
@interface AppDelegate : UIResponder <UIApplicationDelegate>
@property(nonatomic, strong) UIWindow* window;
@end

@implementation AppDelegate

- (void)runUnattended:(const std::vector<Entry>&)entries
                extra:(const std::vector<std::string>&)extra {
    UIApplication.sharedApplication.idleTimerDisabled = YES;
    const std::vector<Entry> todo = entries;
    const std::vector<std::string> args = extra;
    StartRunner(^{
        int failed = 0;
        for (const auto& e : todo) {
            failed += RunEntry(e, args) != 0;
        }
        std::printf("[dressi-ios] finished, %d failed\n", failed);
        std::fflush(stdout);
        exit(failed != 0);
    });
}

- (BOOL)application:(UIApplication*)app
        didFinishLaunchingWithOptions:(NSDictionary*)opts {
    InstallLogSinks();
    chdir(DocumentsDir().c_str());
    self.window = [[UIWindow alloc] initWithFrame:UIScreen.mainScreen.bounds];
    self.window.rootViewController = [[UINavigationController alloc]
            initWithRootViewController:[ExampleListController new]];
    [self.window makeKeyAndVisible];

    // iOS forbids GPU work in the background: stop instead of losing the
    // device (examples exit through their normal teardown).
    [NSNotificationCenter.defaultCenter
            addObserverForName:UIApplicationDidEnterBackgroundNotification
                        object:nil
                         queue:nil
                    usingBlock:^(NSNotification*) {
                        g_cancel = true;
                    }];

    std::vector<Entry> todo;
    std::vector<std::string> extra;
    bool in_extra = false;
    const auto entries = MakeEntries();
    for (NSString* a in NSProcessInfo.processInfo.arguments) {
        const std::string arg = a.UTF8String;
        if (in_extra) {
            extra.push_back(arg);
        } else if (arg == "--") {
            in_extra = true;
        } else if (arg == "--run-all") {
            todo = entries;
        } else if (arg.rfind("--run=", 0) == 0) {
            for (const auto& e : entries) {
                if (e.label == arg.substr(6)) {
                    todo.push_back(e);
                }
            }
        }
    }
    if (!todo.empty()) {
        [self runUnattended:todo extra:extra];
    }
    return YES;
}

@end

int main(int argc, char* argv[]) {
    @autoreleasepool {
        return UIApplicationMain(argc, argv, nil,
                                 NSStringFromClass(AppDelegate.class));
    }
}
