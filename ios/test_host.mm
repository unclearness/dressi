// iOS gtest host: an app is the only thing that runs on a device, and an app
// that does not bring up UIKit is killed by the launch watchdog. Runs every
// suite on a large-stack thread and exits with the gtest result; pass
// --gtest_* launch arguments as usual. Output goes to stdout (devicectl
// --console / simctl launch --console-pty) and Documents/gtest.log.
#import <UIKit/UIKit.h>
#include <gtest/gtest.h>
#include <unistd.h>

#include <cstdio>
#include <string>
#include <vector>

@interface TestHostDelegate : UIResponder <UIApplicationDelegate>
@property(nonatomic, strong) UIWindow* window;
@end

@implementation TestHostDelegate

- (BOOL)application:(UIApplication*)app
        didFinishLaunchingWithOptions:(NSDictionary*)opts {
    self.window = [[UIWindow alloc] initWithFrame:UIScreen.mainScreen.bounds];
    UIViewController* vc = [UIViewController new];
    vc.view.backgroundColor = UIColor.systemBackgroundColor;
    UILabel* label = [[UILabel alloc] initWithFrame:vc.view.bounds];
    label.text = @"Running dressi tests...";
    label.textAlignment = NSTextAlignmentCenter;
    [vc.view addSubview:label];
    self.window.rootViewController = vc;
    [self.window makeKeyAndVisible];
    app.idleTimerDisabled = YES;

    NSThread* thread = [[NSThread alloc] initWithBlock:^{
        NSString* docs = NSSearchPathForDirectoriesInDomains(
                NSDocumentDirectory, NSUserDomainMask, YES).firstObject;
        chdir(docs.UTF8String);
        std::vector<std::string> args;
        for (NSString* a in NSProcessInfo.processInfo.arguments) {
            args.emplace_back(a.UTF8String);
        }
        std::vector<char*> argv;
        for (auto& a : args) {
            argv.push_back(a.data());
        }
        int argc = int(argv.size());
        testing::InitGoogleTest(&argc, argv.data());
        testing::GTEST_FLAG(output) = "xml:gtest.xml";
        const int rc = RUN_ALL_TESTS();
        std::printf("[dressi-ios] gtest exit=%d\n", rc);
        std::fflush(stdout);
        exit(rc);
    }];
    thread.stackSize = 64 << 20;  // secondary threads default to 512 KiB
    [thread start];
    return YES;
}

@end

int main(int argc, char* argv[]) {
    @autoreleasepool {
        return UIApplicationMain(argc, argv, nil,
                                 NSStringFromClass(TestHostDelegate.class));
    }
}
